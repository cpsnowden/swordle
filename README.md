# sign-game-server

## Deployed Environments

1. API for deployed server https://sign-game-server-yckhsn477a-uc.a.run.app/docs

### GCP

1. To see the GCP resources, ensure you are a member of https://groups.google.com/g/lewagonmelbourne/members
2. CLoud Build Console - https://console.cloud.google.com/cloud-build/dashboard?project=wagon-bootcamp-374809
3. CLoud Run Console - https://console.cloud.google.com/run?project=wagon-bootcamp-374809

## Developing Locally

#### Create VENV

```bash
pyenv virtualenv sign-game-server
pyenv activate sign-game-server
pyenv local sign-game-server
```

#### Install Requirements

```bash
pip install -r requirements.txt
```

#### Start Local Fast API

```bash
make run
```
