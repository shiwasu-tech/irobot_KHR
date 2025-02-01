#include "rcb4.h"

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>

rcb4_connection* con; // Connection to the robot
rcb4_comm* comm = NULL; // Command to be sent

#define MSG_KEY 1234
#define MSG_SIZE 256

struct msg_buffer {
    long msg_type;
    char msg_text[MSG_SIZE];
} message;

void deinit() {
    rcb4_command_delete(comm);
    rcb4_deinit(con);
    
    printf("Exit correctly.\n");
}

int main(int argc, char *argv[]) {
    uint8_t buffer[256]; // Data from the robot
    int len; // Length of the data
    char array[22][2]; // 22 rows, 2 columns array

    // Message queue setup
    int msgid = msgget(MSG_KEY, 0666 | IPC_CREAT);
    if (msgid == -1) {
        perror("msgget");
        return -1;
    }

    printf("Connecting to the robot\n");
    con = rcb4_init("/dev/ttyUSB0");
    if (!con) return -1;
    atexit(deinit);
    
    printf("Ping: %d\n", rcb4_command_ping(con));
    
    printf("Reading system configuration (RAM 0x0000).\n");
    comm = rcb4_command_create(RCB4_COMM_MOV);
    rcb4_command_set_src_ram(comm, 0x0000, 2);
    rcb4_command_set_dst_com(comm);
    if ((len = rcb4_send_command(con, comm, buffer)) < 0)
        return -1;
    
    if (len >= 2) printf("Configuration word = 0x%04X.\n", *(uint16_t*)buffer);
    else printf("Could not read the configuration word correctly.\n");
		
		while (1) {
			// Receive message from message queue
			if (msgrcv(msgid, &message, sizeof(message.msg_text), 1, 0) == -1) {
				perror("msgrcv");
				return -1;
			}

			// Convert received string to 22x2 array
			int index = 0;
			for (int i = 0; i < 22; i++) {
				for (int j = 0; j < 2; j++) {
					if (index < strlen(message.msg_text)) {
						array[i][j] = message.msg_text[index++];
					} else {
						array[i][j] = '\0'; // Fill remaining cells with null character
					}
				}
			}

			// Print the array for verification
			for (int i = 0; i < 22; i++) {
				printf("%c %c\n", array[i][0], array[i][1]);
			}
		}

    return 0;
}