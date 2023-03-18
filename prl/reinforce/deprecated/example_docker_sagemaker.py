import argparse

parser = argparse.ArgumentParser()
parser.add_argument("train")
args = parser.parse_args()

if __name__ == '__main__':
    for i in range(10):
        print('Hello Sagemaker and Docker!')
