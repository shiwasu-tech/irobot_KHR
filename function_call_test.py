import ctypes

libc = ctypes.cdll.LoadLibrary("librcb4/func.so")

def main():
    libc.init()
    


if __name__ == "__main__":
    main()
