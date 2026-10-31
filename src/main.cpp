#include "log.h"
#include "config.h"
#include "router.h"
#include "ui.h"

// ───────── 入口：打印启动徽标，初始化日志与配置，然后启动路由服务 ─────────
int main() {
    show_logo();
    init_logger();
    init_config();
    return run_router();
}
