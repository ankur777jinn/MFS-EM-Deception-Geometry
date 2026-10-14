"""
run_all_c9.py — Master runner for the complete C9 control suite.
Runs Part 1 (Inspection & Sanity) followed by Part 2 (Disentanglement & Cost-Matched Controls).
"""
import os
import subprocess
import sys

def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p1 = os.path.join(root, "scripts", "controls", "C9_inspection_and_sanity.py")
    p2 = os.path.join(root, "scripts", "controls", "C9_disentanglement_and_cost_matched.py")

    print("=" * 80)
    print("LAUNCHING FULL C9 CONTROL SUITE")
    print("=" * 80)

    print("\n>>> STEP 1/2: Running C9 Inspection and Sanity Controls...")
    res1 = subprocess.run([sys.executable, p1], check=True)
    if res1.returncode != 0:
        print("ERROR in Part 1! Aborting.")
        sys.exit(res1.returncode)

    print("\n>>> STEP 2/2: Running C9 Disentanglement and Cost-Matched Controls...")
    res2 = subprocess.run([sys.executable, p2], check=True)
    if res2.returncode != 0:
        print("ERROR in Part 2! Aborting.")
        sys.exit(res2.returncode)

    print("\n" + "=" * 80)
    print("ALL C9 CONTROLS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
