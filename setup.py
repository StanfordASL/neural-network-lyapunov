import os

from os.path import dirname, abspath, isdir


def main():
    cur_dir = dirname(abspath(__file__))
    # Now add this folder to PYTHONPATH
    setup_script = cur_dir + "/config/setup_environments.sh"
    if not isdir(cur_dir+"/config"):
        print("create config folder")
        os.mkdir(cur_dir+"/config")
    with open(setup_script, "a") as f:
        f.write(f"""
        export PYTHONPATH={cur_dir}:${{PYTHONPATH}}
        """)
    print("""
    To use neural_network_lyapunov functionality:
        source ./config/setup_environments.sh
    """)


if __name__ == "__main__":
    main()
