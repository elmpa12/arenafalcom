from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    from botocore.exceptions import ClientError, WaiterError  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - dependency optional para --help
    ClientError = WaiterError = None  # type: ignore

from .providers import InstanceMeta

DEFAULT_AMI = "ami-053b0d53c279acc90"  # Ubuntu 22.04 with NVIDIA drivers available in us-east-1
DEFAULT_INSTANCE_TYPE = "g5.xlarge"
DEFAULT_SECURITY_GROUP_NAME = "botscalp-gpu-ssh"
DEFAULT_SSH_CIDRS = ("0.0.0.0/0",)
DEFAULT_VOLUME_SIZE = 200
DEFAULT_PROVIDER_NAME = "aws"
DEFAULT_CLOUD_INIT_PATH = Path(__file__).with_name("default_cloud_init.yaml")


def _lazy_boto3():
    import importlib

    try:
        return importlib.import_module("boto3")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("Dependência boto3 ausente. Instale com 'pip install boto3'.") from exc


def _ensure_botocore() -> None:
    global ClientError, WaiterError
    if ClientError is None or WaiterError is None:  # type: ignore[truthy-bool]
        try:
            from botocore.exceptions import ClientError as _ClientError, WaiterError as _WaiterError
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise SystemExit("Dependência botocore ausente. Instale com 'pip install boto3'.") from exc
        ClientError = _ClientError  # type: ignore[assignment]
        WaiterError = _WaiterError  # type: ignore[assignment]


@dataclass
class AWSLaunchOptions:
    region: str
    instance_type: str = DEFAULT_INSTANCE_TYPE
    ami: str = DEFAULT_AMI
    key_name: Optional[str] = None
    name: str = "BotScalp-GPU"
    spot: bool = False
    max_price: Optional[str] = None
    volume_size: int = DEFAULT_VOLUME_SIZE
    subnet_id: Optional[str] = None
    security_group_id: Optional[str] = None
    ssh_cidrs: Sequence[str] = DEFAULT_SSH_CIDRS
    cloud_init: Optional[str] = None
    wait: bool = True
    tags: Optional[Dict[str, str]] = None


class AWSProvider:
    """Provision BotScalp GPU instances on AWS EC2."""

    def __init__(self) -> None:
        self._session_cache: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Session helpers
    def session(self, region: str):
        if region not in self._session_cache:
            boto3 = _lazy_boto3()
            self._session_cache[region] = boto3.Session(region_name=region)
        return self._session_cache[region]

    # ------------------------------------------------------------------
    def reuse(
        self,
        *,
        region: str,
        name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        states: Sequence[str] = ("pending", "running"),
    ) -> Optional[InstanceMeta]:
        session = self.session(region)
        ec2 = session.resource("ec2")
        filters: List[Dict[str, Sequence[str]]] = []
        if states:
            filters.append({"Name": "instance-state-name", "Values": list(states)})
        if name:
            filters.append({"Name": "tag:Name", "Values": [name]})
        for key, value in (tags or {}).items():
            filters.append({"Name": f"tag:{key}", "Values": [value]})

        candidates = list(ec2.instances.filter(Filters=filters))
        if not candidates:
            return None
        # pick newest running instance first
        candidates.sort(key=lambda inst: (inst.launch_time or datetime.min), reverse=True)
        inst = candidates[0]
        inst.load()
        return self._extract_metadata(inst, region)

    # ------------------------------------------------------------------
    def launch(self, **kwargs) -> InstanceMeta:
        _ensure_botocore()
        options = AWSLaunchOptions(**kwargs)
        session = self.session(options.region)
        ec2_client = session.client("ec2")
        ec2_resource = session.resource("ec2")

        subnet_id = options.subnet_id or self._find_default_subnet(ec2_client)
        security_group_id = options.security_group_id or self._ensure_security_group(
            ec2_client,
            subnet_id,
            options.ssh_cidrs or DEFAULT_SSH_CIDRS,
        )

        tags = {"Name": options.name, "Project": "BotScalp"}
        if options.tags:
            tags.update(options.tags)

        block_device_mappings = [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": int(options.volume_size),
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ]

        network_interfaces = [
            {
                "DeviceIndex": 0,
                "AssociatePublicIpAddress": True,
                "SubnetId": subnet_id,
                "Groups": [security_group_id],
            }
        ]

        user_data = options.cloud_init or self._default_cloud_init()

        instance_kwargs = dict(
            ImageId=options.ami,
            InstanceType=options.instance_type,
            KeyName=options.key_name,
            MinCount=1,
            MaxCount=1,
            NetworkInterfaces=network_interfaces,
            BlockDeviceMappings=block_device_mappings,
            TagSpecifications=[
                {"ResourceType": "instance", "Tags": _to_aws_tags(tags)},
                {"ResourceType": "volume", "Tags": _to_aws_tags(tags)},
            ],
            UserData=user_data,
        )

        if options.spot:
            instance_kwargs["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            }
            if options.max_price:
                instance_kwargs["InstanceMarketOptions"]["SpotOptions"]["MaxPrice"] = options.max_price

        try:
            response = ec2_client.run_instances(**instance_kwargs)
        except ClientError as exc:  # pragma: no cover - depends on AWS
            raise SystemExit(f"Erro ao solicitar instância: {exc}") from exc

        instance_id = response["Instances"][0]["InstanceId"]
        instance = ec2_resource.Instance(instance_id)

        if options.wait:
            self._wait_until_ready(instance)
        else:
            instance.load()

        return self._extract_metadata(instance, options.region, extra_tags=tags)

    def refresh(self, meta: InstanceMeta, *, wait: bool = False) -> InstanceMeta:
        session = self.session(meta.region)
        instance = session.resource("ec2").Instance(meta.instance_id)
        if wait:
            self._wait_until_ready(instance)
        else:
            instance.load()
        return self._extract_metadata(instance, meta.region)

    # ------------------------------------------------------------------
    def terminate(self, meta: InstanceMeta) -> None:
        _ensure_botocore()
        session = self.session(meta.region)
        client = session.client("ec2")
        try:
            client.terminate_instances(InstanceIds=[meta.instance_id])
        except ClientError as exc:  # pragma: no cover - AWS response
            raise SystemExit(f"Falha ao encerrar instância {meta.instance_id}: {exc}") from exc

    # ------------------------------------------------------------------
    def _default_cloud_init(self) -> str:
        return DEFAULT_CLOUD_INIT_PATH.read_text(encoding="utf-8")

    def _find_default_subnet(self, client) -> str:
        response = client.describe_subnets(Filters=[{"Name": "default-for-az", "Values": ["true"]}])
        subnets = response.get("Subnets", [])
        if not subnets:
            raise SystemExit("Nenhuma subnet padrão encontrada; use --subnet-id")
        # prefer public subnet
        subnet = sorted(subnets, key=lambda s: s.get("AvailabilityZone", ""))[0]
        return subnet["SubnetId"]

    def _ensure_security_group(self, client, subnet_id: str, cidrs: Sequence[str]) -> str:
        _ensure_botocore()
        subnet = client.describe_subnets(SubnetIds=[subnet_id])["Subnets"][0]
        vpc_id = subnet["VpcId"]

        groups = client.describe_security_groups(Filters=[
            {"Name": "group-name", "Values": [DEFAULT_SECURITY_GROUP_NAME]},
            {"Name": "vpc-id", "Values": [vpc_id]},
        ]).get("SecurityGroups", [])

        if groups:
            sg_id = groups[0]["GroupId"]
        else:
            response = client.create_security_group(
                GroupName=DEFAULT_SECURITY_GROUP_NAME,
                Description="BotScalp GPU SSH",
                VpcId=vpc_id,
            )
            sg_id = response["GroupId"]

        # ensure ingress rule for SSH
        existing = client.describe_security_group_rules(
            Filters=[
                {"Name": "group-id", "Values": [sg_id]},
            ]
        ).get("SecurityGroupRules", [])

        # Filter manually for TCP port 22 ingress
        existing = [
            rule for rule in existing
            if rule.get("IpProtocol") == "tcp"
            and rule.get("FromPort") == 22
            and rule.get("ToPort") == 22
            and not rule.get("IsEgress", False)
        ]

        defined_cidrs = {rule.get("CidrIpv4") for rule in existing if rule.get("CidrIpv4")}
        desired = set(cidrs)
        missing = desired - defined_cidrs
        if missing:
            try:
                client.authorize_security_group_ingress(
                    GroupId=sg_id,
                    IpPermissions=[
                        {
                            "IpProtocol": "tcp",
                            "FromPort": 22,
                            "ToPort": 22,
                            "IpRanges": [{"CidrIp": cidr} for cidr in missing],
                        }
                    ],
                )
            except ClientError as exc:
                if exc.response["Error"].get("Code") != "InvalidPermission.Duplicate":
                    raise
        return sg_id

    def _wait_until_ready(self, instance) -> None:
        _ensure_botocore()
        instance.wait_until_running()
        instance.reload()
        client = instance.meta.client
        waiter = client.get_waiter("instance_status_ok")
        try:
            waiter.wait(InstanceIds=[instance.id], WaiterConfig={"Delay": 15, "MaxAttempts": 40})
        except WaiterError as exc:  # pragma: no cover - depends on AWS
            raise SystemExit(f"Instância {instance.id} não ficou pronta: {exc}") from exc

    def _extract_metadata(self, instance, region: str, extra_tags: Optional[Dict[str, str]] = None) -> InstanceMeta:
        instance.load()
        tags = extra_tags.copy() if extra_tags else {}
        for tag in instance.tags or []:
            tags[tag.get("Key")] = tag.get("Value")

        return InstanceMeta(
            provider=DEFAULT_PROVIDER_NAME,
            region=region,
            instance_id=instance.id,
            state=(instance.state or {}).get("Name", "unknown"),
            public_ip=getattr(instance, "public_ip_address", None),
            private_ip=getattr(instance, "private_ip_address", None),
            name=tags.get("Name"),
            launch_time=getattr(instance, "launch_time", None).isoformat() if getattr(instance, "launch_time", None) else None,
            tags=tags,
        )


def _to_aws_tags(mapping: Dict[str, str]) -> List[Dict[str, str]]:
    return [{"Key": key, "Value": value} for key, value in mapping.items()]
