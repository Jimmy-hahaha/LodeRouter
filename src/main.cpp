#include "log.h"
#include "config.h"
#include "router.h"

int main() {
    init_logger();
    init_config();
    return run_router();
}
