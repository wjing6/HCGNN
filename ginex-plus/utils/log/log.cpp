#include "log.h"

#ifdef USE_LOG

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <ctime>

#include <mutex>
#include <cmath>

#include "../timer.h"

using namespace std;
using namespace std::chrono;

std::mutex global_log_mutex;
Timer timer;

static struct {
    void *udata;
    log_LockFn lock;
    FILE *fp;
    int level;
    int quiet;
} L;


static const char *level_names[] = {
        "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"
};
#ifndef LOG_USE_COLOR
#define LOG_USE_COLOR
#endif
#ifdef LOG_USE_COLOR
static const char *level_colors[] = {
        "\x1b[94m", "\x1b[36m", "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[35m"
};
#endif


static void lock(void) {
    if (L.lock) {
        L.lock(L.udata, 1);
    }
}


static void unlock(void) {
    if (L.lock) {
        L.lock(L.udata, 0);
    }
}


void log_set_udata(void *udata) {
    L.udata = udata;
}


void log_set_lock(log_LockFn fn) {
    L.lock = fn;
}


void log_set_fp(FILE *fp) {
    L.fp = fp;
}


void log_set_level(int level) {
    L.level = level;
}


void log_set_quiet(int enable) {
    L.quiet = enable ? 1 : 0;
}


void log_log(int level, const char *file, const char *func, int line, const char *fmt, ...) {
    if (level < L.level) {
        return;
    }

    using namespace std::chrono;
    auto elapsed_time = timer.elapsed();
    {
        unique_lock<mutex> lock_global(global_log_mutex);
        /* Acquire lock */
        lock();

        /* Get current time */
        time_t t = time(nullptr);
        struct tm *lt = localtime(&t);

        /* Log to stderr */
        if (!L.quiet) {
            va_list args;
            char buf[64];
            buf[strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", lt)] = '\0';
#ifdef LOG_USE_COLOR
            fprintf(
                    stderr, "%s %s%-5s (et: %.6lf s)\x1b[0m\x1b[36m (func: %s) \x1b[0m \x1b[90m%s:%d:\x1b[0m ",
                    buf, level_colors[level], level_names[level],
                    elapsed_time, func, file, line);
#else
            fprintf(stderr, "%s %-5s (et: %.6lf s) %s:%d: ", buf, level_names[level],
                    elapsed_time,  func, file, line);
#endif
            va_start(args, fmt);
            vfprintf(stderr, fmt, args);
            va_end(args);
            fprintf(stderr, "\n");
        }

        /* Log to file */
        if (L.fp) {
            va_list args;
            char buf[32];
            buf[strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", lt)] = '\0';
            fprintf(L.fp, "%s %-5s (et: %.6lf s) (func: %s) %s:%d: ", buf, level_names[level],
                    elapsed_time, func, file, line);
            va_start(args, fmt);
            vfprintf(L.fp, fmt, args);
            va_end(args);
            fprintf(L.fp, "\n");
        }

        /* Release lock */
        unlock();
    }
}

#endif