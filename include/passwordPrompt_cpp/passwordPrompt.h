#ifndef PASSWORD_PROMPT_H
#define PASSWORD_PROMPT_H

#include <string>

class PasswordPrompt {
public:
    static std::string getPassword(const std::string& prompt = "Enter password: ");
};

#endif