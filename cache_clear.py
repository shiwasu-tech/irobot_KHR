import warnings
import sysv_ipc
warnings.simplefilter('ignore', FutureWarning)

# Message queue ID
QUE_ID = 1234

def clear_message_queue(queue_id):
    '''メッセージキューのキャッシュをクリアする'''
    try:
        # メッセージキューに接続
        mq = sysv_ipc.MessageQueue(queue_id)
        
        # メッセージキュー内のすべてのメッセージを削除
        while True:
            try:
                mq.receive(block=False)
            except sysv_ipc.BusyError:
                break
    except sysv_ipc.ExistentialError:
        print(f"Message queue with ID {queue_id} does not exist.")

if __name__ == '__main__':
    clear_message_queue(QUE_ID)