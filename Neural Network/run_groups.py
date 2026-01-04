import subprocess
import time
import sys
import os

GROUPS = [
    [0, 9, 18, 27, 36]
    # .add Group 5, 6, 7, 8, 9 ...
]

SECONDS_PER_GROUP = 300 
SERVER_ADDRESS = "127.0.0.1:8080"

# Find paths automatically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")

def main():
    try:
        for i, user_ids in enumerate(GROUPS):
            print(f" STARTING SESSION FOR GROUP {i} ")

            # START SERVER 
            print(" Launching Server.")
            server_process = subprocess.Popen(["python", "server/server_flwr.py"])
            
            # Wait for server to be ready
            time.sleep(10) 

            # START CLIENTS
            current_processes = []
            print(f"Launching Clients: {user_ids}")

            for user_id in user_ids:
                client_id = f"user_{user_id}"
                filename = f"dataset_user_{user_id}_train.csv"
                csv_path = os.path.join(DATA_DIR, filename)

                if not os.path.exists(csv_path):
                    print(f" Missing file: {filename}")
                    continue

                cmd = [
                    sys.executable, "client_flwr.py",
                    "--server_address", SERVER_ADDRESS,
                    "--client_id", client_id,
                    "--csv_path", csv_path
                ]
                
                p = subprocess.Popen(cmd)
                current_processes.append(p)

            # TRAIN
            print(f" Group {i} is training for {SECONDS_PER_GROUP}s.")
            time.sleep(SECONDS_PER_GROUP)

            # STOP EVERYTHING
            print(f" Stopping Group {i}.")
            for p in current_processes:
                p.terminate()
            
            print(" Stopping Server.")
            server_process.terminate()
            
            # Wait for ports to clear
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n Interrupted.")
        try:
            server_process.terminate()
        except:
            pass

if __name__ == "__main__":
    main()

