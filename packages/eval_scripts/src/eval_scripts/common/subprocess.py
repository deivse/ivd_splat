import subprocess
import sys


def subprocess_run_tee_stderr(*args, **kwargs) -> tuple[int, str]:
    with subprocess.Popen(*args, stderr=subprocess.PIPE, text=False, **kwargs) as p:
        stderr_output = bytes()
        while True:
            text_bytes = p.stderr.read1()
            if not text_bytes:
                break
            stderr_output += text_bytes
            sys.stderr.buffer.write(text_bytes)
            sys.stderr.buffer.flush()

        p.wait()
        rc = p.returncode
        return rc, stderr_output.decode("utf-8")
