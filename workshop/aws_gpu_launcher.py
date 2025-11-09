#!/usr/bin/env python3
"""Provision temporary AWS GPU instances for BotScalp."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

from tools.aws_provider import (
    AWSProvider,
    DEFAULT_AMI,
    DEFAULT_CLOUD_INIT_PATH,
    DEFAULT_INSTANCE_TYPE,
    DEFAULT_SSH_CIDRS,
    DEFAULT_VOLUME_SIZE,
)
from tools.providers import DEFAULT_METADATA_PATH, InstanceMeta, save_metadata


def parse_tags(items: Iterable[str]) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Tag inválida: '{item}' (use CHAVE=VALOR)")
        key, value = item.split("=", 1)
        tags[key.strip()] = value.strip()
    return tags


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="us-east-1", help="Região AWS")
    ap.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    ap.add_argument("--ami", default=DEFAULT_AMI, help="AMI a ser usada")
    ap.add_argument("--key-name", required=True, help="Par de chaves EC2 já existente")
    ap.add_argument("--name", default="BotScalp-GPU", help="Tag Name")
    ap.add_argument("--tag", action="append", default=[], help="Tag adicional no formato K=V", dest="tags")
    ap.add_argument("--spot", action="store_true", help="Solicita instância spot")
    ap.add_argument("--max-price", help="Preço máximo da spot (opcional)")
    ap.add_argument("--volume-size", type=int, default=DEFAULT_VOLUME_SIZE, help="Tamanho do disco root em GB")
    ap.add_argument("--subnet-id", help="Subnet específica")
    ap.add_argument("--security-group-id", help="Security Group já existente")
    ap.add_argument(
        "--ssh-cidr",
        action="append",
        default=list(DEFAULT_SSH_CIDRS),
        help="CIDR liberado para SSH (padrão 0.0.0.0/0)",
        dest="ssh_cidrs",
    )
    ap.add_argument("--cloud-init", help="Arquivo YAML customizado de cloud-init")
    ap.add_argument("--no-wait", action="store_true", help="Não aguarda até a instância ficar pronta")
    ap.add_argument("--reuse", action="store_true", help="Reutiliza instância com as mesmas tags")
    ap.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH), help="Onde salvar os metadados")
    return ap.parse_args(argv)


def load_cloud_init(path: Optional[str]) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return DEFAULT_CLOUD_INIT_PATH.read_text(encoding="utf-8")


def describe(meta: InstanceMeta) -> str:
    ip = meta.public_ip or meta.private_ip or "<sem IP>"
    return (
        f"Instância {meta.instance_id} ({meta.state})\n"
        f"Região: {meta.region}\n"
        f"IP: {ip}"
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    provider = AWSProvider()

    tags = parse_tags(args.tags)
    cloud_init = load_cloud_init(args.cloud_init)

    if args.reuse:
        reused = provider.reuse(region=args.region, name=args.name, tags=tags)
    else:
        reused = None

    if reused:
        wait = not args.no_wait
        meta = provider.refresh(reused, wait=wait)
        print(f"Reutilizando instância existente: {meta.instance_id} ({meta.state})")
    else:
        meta = provider.launch(
            region=args.region,
            instance_type=args.instance_type,
            ami=args.ami,
            key_name=args.key_name,
            name=args.name,
            spot=args.spot,
            max_price=args.max_price,
            volume_size=args.volume_size,
            subnet_id=args.subnet_id,
            security_group_id=args.security_group_id,
            ssh_cidrs=args.ssh_cidrs,
            cloud_init=cloud_init,
            wait=not args.no_wait,
            tags=tags,
        )

    save_metadata(meta, Path(args.metadata))
    print(describe(meta))
    if meta.public_ip:
        print(f"ssh -i ~/.ssh/{args.key_name}.pem ubuntu@{meta.public_ip}")
    else:
        print("A instância ainda não possui IP público. Consulte o console AWS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
