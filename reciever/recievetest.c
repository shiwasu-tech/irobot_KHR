#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>

#define MSG_KEY 1234
#define MSG_SIZE 256

struct msg_buffer {
    long msg_type;
    char msg_text[MSG_SIZE];
} message;

int main() {
    // Message queue setup
    int msgid = msgget(MSG_KEY, 0666 | IPC_CREAT);
    if (msgid == -1) {
        perror("msgget");
        return -1;
    }

    while (1) {
        // Receive message from message queue
        if (msgrcv(msgid, &message, sizeof(message.msg_text), 1, 0) == -1) {
            perror("msgrcv");
            return -1;
        }

        // Print the received message
        printf("Received message:\n%s\n", message.msg_text);

        // Convert received string to 22x2 array
        float array[22][2];
        int row = 0, col = 0;
        char *token = strtok(message.msg_text, ",\n");
        while (token != NULL && row < 22) {
            array[row][col] = atof(token);
            col++;
            if (col == 2) {
                col = 0;
                row++;
            }
            token = strtok(NULL, ",\n");
        }

        // Print the array for verification
        printf("Converted array:\n");
        for (int i = 0; i < 22; i++) {
            printf("%f %f\n", array[i][0], array[i][1]);
        }

        // Extract and print rows where the first column has a value of 2
        printf("Rows where the first column is 2:\n");
        for (int i = 0; i < 22; i++) {
            if ((int)array[i][0] == 2) {
                printf("%f %f\n", array[i][0], array[i][1]);
            }
        }
    }

    return 0;
}
