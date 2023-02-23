#include <stdio.h>
void encrypt_caesar(char plaintext[], int shift);
void decrypt_caesar(char ciphertext[], int shift); 

int main() {
    char message[] = "Hello, world!";
    int shift = 3; // 얼마만큼 알파벳의 순서 이동이 이루어질 것이냐 : 이 숫자를 변경하면 결과도 바뀜

    printf("Original message: %s\n", message);

    encrypt_caesar(message, shift);
    printf("Encrypted message: %s\n", message);

    decrypt_caesar(message, shift);
    printf("Decrypted message: %s\n", message);

    return 0;
}

void encrypt_caesar(char plaintext[], int shift) {
    int i;
    for (i = 0; plaintext[i] != '\0'; i++) {
        if (plaintext[i] >= 'a' && plaintext[i] <= 'z') {
            plaintext[i] = (plaintext[i] - 'a' + shift) % 26 + 'a';
        } else if (plaintext[i] >= 'A' && plaintext[i] <= 'Z') {
            plaintext[i] = (plaintext[i] - 'A' + shift) % 26 + 'A';
        }
    }
}

void decrypt_caesar(char ciphertext[], int shift) {
    encrypt_caesar(ciphertext, 26 - shift);
}