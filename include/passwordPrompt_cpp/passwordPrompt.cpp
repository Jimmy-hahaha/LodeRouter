// Written by Patrick S. Avery -- 2015
// Modified to display '*' as placeholder for each input character
// Returns the password

#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

#include "passwordPrompt.h"

std::string PasswordPrompt::getPassword(const std::string& prompt)
{
    std::string password;
    char ch = 0;

#ifdef _WIN32
    // Windows
    std::cout << prompt<<<< std::flush;
    while (true) {
        ch = _getch();       
        if (ch == '\r' || ch == '\n') 
            break;
        else if (ch == '\b' || ch == 127) { 
            if (!password.empty()) {
                password.pop_back();
                std::cout << "\b \b";
            }
        } else {
            password.push_back(ch);
            std::cout << '*'; 
        }
    }
    std::cout << std::endl;

#else
    // Unix
    termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    std::cout << prompt << std::flush;
    while (read(STDIN_FILENO, &ch, 1) == 1) {
        if (ch == '\n' || ch == '\r')
            break;
        else if (ch == 127 || ch == '\b') {
            if (!password.empty()) {
                password.pop_back();
                std::cout << "\b \b";
                std::cout.flush();
            }
        } else {
            password.push_back(ch);
            std::cout << '*';
            std::cout.flush();
        }
    }

    // Clean up
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    std::cout << std::endl;
#endif

    return password;
}