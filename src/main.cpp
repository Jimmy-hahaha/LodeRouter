#include "log.h"
#include "config.h"
#include "router.h"

// ───────── 入口：初始化日志与配置，然后启动路由服务 ─────────
int main() {
    init_logger();
    init_config();
    return run_router();
}
