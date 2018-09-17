import _thread
import time


def CounterDown(ThreadName):
    count = 10
    while (count >= 0):
        print("{} executes -{}".format(ThreadName, count))
        count = count - 1
        time.sleep(2)


def main():
    try:
        _thread.start_new_thread(CounterDown, ("First",))
        _thread.start_new_thread(CounterDown, ("Second",))
        print("I am main Thread")
    except:
        print("Error: Failed to start the thread!!")

    while True:
        pass


if __name__ == '__main__':
    main()
