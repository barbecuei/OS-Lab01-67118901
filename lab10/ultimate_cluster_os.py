
# ultimate_cluster_os.py

import threading
import time
import os
import queue


class AIClusterOS:
    def __init__(self, total_ram_gb, num_gpus):
        # 1. System Resources
        self.total_ram_gb = total_ram_gb
        self.available_ram_gb = total_ram_gb

        self.ram_lock = threading.Lock()

        self.gpu_locks = {
            i: threading.Lock() for i in range(num_gpus)
        }

        self.gpu_status = {
            i: "IDLE" for i in range(num_gpus)
        }

        # 2. OS Queues and State
        self.job_queue = queue.Queue()
        self.active_jobs = []
        self.workers = []
        self.is_running = True

        # 3. Start Dashboard and Scheduler Threads
        self.dash_thread = threading.Thread(
            target=self._dashboard_loop,
            daemon=True
        )
        self.dash_thread.start()

        self.scheduler_thread = threading.Thread(
            target=self._os_scheduler_loop
        )
        self.scheduler_thread.start()

    def _dashboard_loop(self):
        """Display simulated cluster resources."""

        while self.is_running:
            time.sleep(1.5)

            with self.ram_lock:
                available = self.available_ram_gb
                gpu_str = " | ".join(
                    f"GPU {i}: {self.gpu_status[i]}"
                    for i in self.gpu_locks
                )
                active = len(self.active_jobs)

            print("\n" + "=" * 50)
            print(
                f"[LIVE DASHBOARD] RAM Available: "
                f"{available}/{self.total_ram_gb} GB"
            )
            print(f"[LIVE DASHBOARD] {gpu_str}")
            print(
                f"[LIVE DASHBOARD] Queue Size: "
                f"{self.job_queue.qsize()} | Active Jobs: {active}"
            )
            print("=" * 50 + "\n")

    def submit_job(self, job_name, dataset_path,
                   req_ram, req_gpus, duration):

        self.job_queue.put(
            (job_name, dataset_path, req_ram, req_gpus, duration)
        )

        print(f"[API] Submitted: {job_name} -> Queued.")

    def _os_scheduler_loop(self):
        """Start a worker thread for each queued job."""

        while self.is_running or not self.job_queue.empty():
            try:
                job_data = self.job_queue.get(timeout=0.5)

                worker = threading.Thread(
                    target=self._execute_job,
                    args=job_data
                )

                self.workers.append(worker)
                worker.start()

            except queue.Empty:
                continue

    def _execute_job(self, job_name, dataset_path,
                     req_ram, req_gpus, duration):

        allocated_ram = False
        acquired_gpus = []

        try:
            # 1. FILE SYSTEM CHECK (Lab 9)
            if not os.path.exists(dataset_path):
                print(
                    f"[{job_name}] FAILED: "
                    f"Dataset '{dataset_path}' not found."
                )
                return

            with self.ram_lock:
                self.active_jobs.append(job_name)

            # 2. MEMORY MANAGEMENT (Labs 5/6)
            print(f"[{job_name}] Waiting for {req_ram}GB RAM...")

            while True:
                with self.ram_lock:
                    if self.available_ram_gb >= req_ram:
                        self.available_ram_gb -= req_ram
                        allocated_ram = True
                        break

                time.sleep(0.5)

            print(f"[{job_name}] Allocated {req_ram}GB RAM.")

            # 3. DEADLOCK AVOIDANCE (Lab 4)
            sorted_gpus = sorted(req_gpus)

            if sorted_gpus:
                print(
                    f"[{job_name}] Waiting for GPUs "
                    f"{sorted_gpus}..."
                )

            for gpu in sorted_gpus:
                self.gpu_locks[gpu].acquire()
                acquired_gpus.append(gpu)

                with self.ram_lock:
                    self.gpu_status[gpu] = f"BUSY ({job_name})"

            # 4. EXECUTION (Lab 7)
            if sorted_gpus:
                print(
                    f"[{job_name}] Acquired GPUs "
                    f"{sorted_gpus}. Running!"
                )
            else:
                print(f"[{job_name}] Running on CPU only!")

            time.sleep(duration)

            print(f"[{job_name}] Finished successfully.")

        finally:
            # 5. RELEASE RESOURCES
            for gpu in reversed(acquired_gpus):
                with self.ram_lock:
                    self.gpu_status[gpu] = "IDLE"

                self.gpu_locks[gpu].release()

            with self.ram_lock:
                if allocated_ram:
                    self.available_ram_gb += req_ram

                if job_name in self.active_jobs:
                    self.active_jobs.remove(job_name)

            self.job_queue.task_done()

    def shutdown(self):
        """Wait for all jobs before shutting down."""

        self.job_queue.join()

        self.is_running = False
        self.scheduler_thread.join()

        for worker in self.workers:
            worker.join()

        self.dash_thread.join()

        print("\n=== Cluster OS Shutdown Gracefully ===")


def main():
    # Create a dummy dataset file
    with open("secure_dataset.csv", "w") as f:
        f.write("dummy data")

    print("=== Booting AI Cluster OS (64GB RAM, 4 GPUs) ===")

    os_system = AIClusterOS(
        total_ram_gb=64,
        num_gpus=4
    )

    # Workload A: Distributed Training
    os_system.submit_job(
        "Workload_A_LLaMA",
        "secure_dataset.csv",
        req_ram=40,
        req_gpus=[2, 1, 0],
        duration=8
    )

    time.sleep(1)

    # Workload B: Data Preprocessing
    os_system.submit_job(
        "Workload_B_Preproc",
        "secure_dataset.csv",
        req_ram=16,
        req_gpus=[],
        duration=6
    )

    time.sleep(1)

    # Workload C: AI Inference
    os_system.submit_job(
        "Workload_C_Infer",
        "secure_dataset.csv",
        req_ram=2,
        req_gpus=[3],
        duration=3
    )

    time.sleep(1)

    # Workload D: Missing Dataset
    os_system.submit_job(
        "Workload_D_Hacker",
        "secret_keys.txt",
        req_ram=1,
        req_gpus=[],
        duration=1
    )

    # Wait for all jobs to finish
    os_system.shutdown()

    # Cleanup
    os.remove("secure_dataset.csv")


if __name__ == "__main__":
    main()
