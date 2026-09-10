import logging
import os
import time
logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Dict, Any
import torch


PROXY_ENV_KEYS = [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
]
MSB_LEFT_COUNT_CHIPS = {"Baihua"}


def _is_proxy_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        "ProxyError" in text
        or "Unable to connect to proxy" in text
        or "Max retries exceeded" in text
        or "TLS/SSL connection has been closed" in text
    )


def _create_task_manager(api: str, disable_proxy: bool):
    try:
        from quark import Task
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The quark SDK is required only for real-machine inference. "
            "It is not needed for normal QVIT_mix2 training. Install the vendor "
            "Quark SDK only if you run with use_real=True."
        ) from exc

    if not api:
        raise ValueError("Missing real-machine API token. Provide --api-token or QUARK_API_TOKEN.")

    if not disable_proxy:
        try:
            return Task(token=api)
        except Exception as e:
            if "token is expired" in str(e):
                raise ValueError("Real-machine API token is expired. Update --api-token or QUARK_API_TOKEN.") from e
            raise

    backup = {k: os.environ.get(k) for k in PROXY_ENV_KEYS}
    try:
        for k in PROXY_ENV_KEYS:
            os.environ.pop(k, None)
        try:
            return Task(token=api)
        except Exception as e:
            if "token is expired" in str(e):
                raise ValueError("Real-machine API token is expired. Update --api-token or QUARK_API_TOKEN.") from e
            raise
    finally:
        for k, v in backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

@dataclass
class Result_Real_Machine:
    # ========== measurement ==========
    count: Dict[str, int]
    corrected: Dict[str, Any]

    # ========== backend ==========
    chip: str
    shots: int

    # ========== circuits ==========
    circuit: str
    transpiled: str
    qlisp: str

    # ========== runtime ==========
    tid: int
    error: str
    status: str

    # ========== time ==========
    created: str
    finished: str
    
    def get_exp_res(self, num_qubits: int):

        vec = torch.zeros(num_qubits, dtype=torch.float32)
        total = sum(self.count.values())
        if total == 0:
            return vec
        bitstring_left = "MSB" if str(self.chip) in MSB_LEFT_COUNT_CHIPS else "LSB"

        for bitstring, c in self.count.items():
            p = c / total
            bits = bitstring[::-1] if bitstring_left == "MSB" else bitstring

            # 閬嶅巻姣忎竴浣?
            for i, bit in enumerate(bits):
                val = 1.0 if bit == "0" else -1.0
                vec[i] += p * val

        return vec

'鐪熸満杩愯鍏ュ彛'
def run_real_machine(
    qasm: str,
    api: str,
    chip: str,
    task_name: str,
    shots: int,
    timeout: float | None = 300,
    time_wait: float = 2.0,
    max_retry_submit: int = 3,
    retry_backoff: float = 3.0,
    disable_proxy: bool = False,
) -> "Result_Real_Machine":

    tmgr = None
    submit_error = None
    for attempt in range(1, max_retry_submit + 1):
        disable_proxy_this_attempt = disable_proxy or (attempt > 1)
        try:
            tmgr = _create_task_manager(api, disable_proxy_this_attempt)
            if disable_proxy_this_attempt:
                logger.info("Task manager created with system proxy disabled on attempt %s", attempt)
            break
        except Exception as e:
            submit_error = e
            if not _is_proxy_error(e) or attempt == max_retry_submit:
                raise
            logger.warning("Task manager creation failed on attempt %s; retrying: %s", attempt, e)
            time.sleep(retry_backoff * attempt)

    if tmgr is None:
        raise RuntimeError(f"Task manager creation failed: {submit_error}")

    task = {
        "chip": chip,
        "name": task_name,
        "circuit": qasm,
        "compile": True
    }
    logger.info("Submitting task with shots=%s", shots)
    # ========== 鎻愪氦浠诲姟 ==========
    start_time = time.time()

    tid = None
    for attempt in range(1, max_retry_submit + 1):
        try:
            tid = tmgr.run(task, shots)
            break
        except Exception as e:
            if attempt == max_retry_submit:
                logger.error("Task submit failed | chip=%s, name=%s, error=%s", chip, task_name, str(e))
                raise
            logger.warning("Task submit failed on attempt %s; retrying: %s", attempt, e)
            time.sleep(retry_backoff * attempt)

    logger.info("Task submitted | name=%s | tid=%s | chip=%s", task_name, tid, chip)

    # ========== 杞浠诲姟鐘舵€?==========
    while True:
        try:
            status = tmgr.status(tid)
        except Exception as e:
            if timeout is not None and timeout > 0 and time.time() - start_time > timeout:
                logger.error("Status query timed out | tid=%s | error=%s", tid, e)
                raise TimeoutError(f"Status query timed out: {tid}")
            logger.warning("Status query failed; retrying | tid=%s | error=%s", tid, e)
            time.sleep(time_wait)
            continue
        # logger.info(f"鈴?浠诲姟鐘舵€?| 浠诲姟ID={tid} | status={status}")

        # ========= 鎻愪氦 =========
        if status == 'Submitted':
            logger.info("Task status=%s", status)

        # ========= 缂栬瘧 =========
        elif status == 'Transpiling':
            logger.info("Task status=%s", status)

        # ========= 瀹屾垚缂栬瘧 =========
        elif status == 'Transpiled':
            logger.info("Task status=%s", status)

        # ========= 鎺掗槦 =========
        elif status == 'Pending':
            logger.info("Task status=%s", status)

        # ========= 杩愯 =========
        elif status == 'Running':
            logger.info("Task status=%s", status)

        # ========= 鍙栨秷 =========
        elif status == 'Cancled':
            logger.error("Task cancelled | status=%s", status)
            break

        # ========= 澶辫触 =========
        elif status == 'Failed':
            logger.error("Task failed | status=%s", status)
            raise RuntimeError(f"Task failed: {tid}, status={status}")

        # ========= 鎴愬姛 =========
        elif status == 'Finished':
            logger.info("Task finished | status=%s", status)
            break

        # ========= 鏈煡鐘舵€?=========
        else:
            logger.warning("Unknown task status=%s", status)

        # ========= 瓒呮椂 =========
        if timeout is not None and timeout > 0 and (time.time() - start_time > timeout):
            logger.error("Task timed out | tid=%s", tid)
            raise TimeoutError(f"Task timed out: {tid}")

        time.sleep(time_wait)

    # ========== 鑾峰彇缁撴灉 ==========
    try:
        res = tmgr.result(tid)
        
        '鍏抽敭璋冭瘯淇℃伅'
        # logger.info(f"馃攳 杩斿洖缁撴灉鐨勫瓧娈? {res.keys()}")
        # logger.info(f"馃攳 res['shots'] = {res.get('shots', 'no shots info')}")
        # logger.info(f"馃攳 type(res['shots']) = {type(res.get('shots'))}")
        # logger.info(f"馃攳 sum(count.values()) = {sum(res.get('count', {}).values())}")
    except Exception as e:
        logger.error("Result fetch failed | tid=%s | error=%s", tid, str(e))
        raise

    logger.info("Result fetched | tid=%s", tid)

    # ========== 瑙ｆ瀽缁撴灉 ==========
    result = Result_Real_Machine(
        count=res.get("count", {}),
        corrected=res.get("corrected", {}),

        chip=res.get("chip", chip),
        shots=res.get('shots', 'Unknow'),

        circuit=res.get("circuit", qasm),
        transpiled=res.get("transpiled", ""),
        qlisp=res.get("qlisp", ""),

        tid=res.get("tid", tid),
        error=res.get("error", ""),
        status=res.get("status", "Unknow"),

        created=res.get("created", ""),
        finished=res.get("finished", "")
    )

    logger.info("Task flow completed | tid=%s | shots=%s | status=%s", tid, result.shots, result.status)

    return result

if __name__=='__main__':
    from generate_qasm import generate_qasm
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s')

    num_qubits = 5
    num_layers = 2

    x_token = torch.randn(num_qubits)

    weights = torch.randn(num_layers + 1, 2*num_qubits)

    qasm = generate_qasm(
        num_qubits=num_qubits,
        x_token=x_token,
        weights=weights
    )

    api1 = os.getenv("QUARK_API_TOKEN")
    res = run_real_machine(qasm = qasm, 
                           api = api1, 
                           chip = 'Dongling', 
                           task_name = 'local_backend', 
                           shots = 10000, 
                           time_wait = 2)
    exp = res.get_exp_res(num_qubits)
    
    print("鈴憋笍 浠诲姟鍒涘缓鏃堕棿:", res.created)
    print("馃弫 浠诲姟瀹屾垚鏃堕棿:", res.finished)
    print("馃搳 娴嬮噺缁撴灉:", res.count)
    print("馃搳 鏈熸湜缁撴灉:", exp)
