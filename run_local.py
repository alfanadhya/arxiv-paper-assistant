"""Run the app on this Windows PC (Python 3.14) for local testing.

    .venv\\Scripts\\python.exe run_local.py

Same as `chainlit run threeofus_main.py -w -h --host 0.0.0.0 --port 8000`, but with nest_asyncio
switched off. Chainlit's CLI calls nest_asyncio.apply(), which breaks anyio on Python 3.14: every
static file (/assets/*.js, custom.js) then fails with a 500 error and the page stays blank.
The app does not need nest_asyncio (it is only used by Chainlit's chat profiles, which we do not use).
Extra arguments are passed on, e.g. `run_local.py --port 8001`. Not needed on the Azure VM.
"""
import sys
import types

sys.modules["nest_asyncio"] = types.SimpleNamespace(apply=lambda *args, **kwargs: None)

from chainlit.cli import cli  # noqa: E402

cli(["run", "threeofus_main.py", "-w", "-h", "--host", "0.0.0.0", "--port", "8000", *sys.argv[1:]],
    prog_name="chainlit")
