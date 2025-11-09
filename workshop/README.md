# botscalp

My Secret

## Exporting the project for download

To create a zip archive with the source files (excluding caches, data folders and other build artefacts), run:

```bash
python tools/archive_project.py --output botscalpv3.zip
```

The command writes `botscalpv3.zip` in the current directory. Use `--exclude-dir` or `--exclude-pattern` for extra filters, e.g. to skip CSV outputs:

```bash
python tools/archive_project.py --exclude-pattern .csv
```

All paths are resolved relative to the repository root by default; pass `--root` if you want to archive a different folder.

## Integrating the GPT gateway with GitHub

The `backend/openai_gateway.py` service does not automatically connect to GitHub.
However, you can trigger it from your repositories by wiring either a webhook or a
GitHub Actions workflow to call the `/api/codex` endpoint exposed by the gateway.

1. **Deploy the gateway** somewhere that your GitHub workflow can reach (for local
   experiments you can use a tunnel solution such as ngrok).
2. **Create a GitHub secret** (for example `GATEWAY_TOKEN`) if you want to protect the
   endpoint with an authentication header before exposing it publicly.
3. **Add a workflow** file such as `.github/workflows/gpt-codex.yml` in your repo:

   ```yaml
   name: Request GPT code suggestion
   on:
     workflow_dispatch:
       inputs:
         prompt:
           description: "Prompt to send to the gateway"
           required: true
   jobs:
     call-gateway:
       runs-on: ubuntu-latest
       steps:
         - name: Call GPT gateway
           run: |
             curl -X POST "$GATEWAY_URL/api/codex" \
               -H "Content-Type: application/json" \
               -H "Authorization: Bearer $GATEWAY_TOKEN" \
               -d "{\"prompt\": \"${{ github.event.inputs.prompt }}\"}"
           env:
             GATEWAY_URL: ${{ secrets.GATEWAY_URL }}
             GATEWAY_TOKEN: ${{ secrets.GATEWAY_TOKEN }}
   ```

With this setup, triggering the workflow from GitHub’s UI will forward the supplied
prompt to your self-hosted gateway, keeping the OpenAI key confined to your server.
