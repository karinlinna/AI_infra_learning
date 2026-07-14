# tenglong-scada-app 内存优化任务设计

## 1. 任务定义

本任务的目标是在**不改变现有业务功能、接口契约、数据库语义和部署拓扑**的前提下，排查并优化 `10.235.236.25` 上的 `tenglong-scada-app`，使其 **Java 进程 RSS/RES 常驻内存稳定低于 600 MB**。

当前阶段只输出任务设计与执行文档，不直接实施优化；后续执行时允许先在本地完成验证，再将产物替换到服务器容器内验证。


## 2. 目标对象

| 项目 | 内容 |
|---|---|
| 应用 | `tenglong-scada-app` |
| 目标机器 | `10.235.236.25` |
| 连接信息 | `root` / `<REDACTED_PASSWORD>` |
| 验收指标 | Java 进程 RSS/RES 常驻内存 `< 600 MB` |
| 允许手段 | JVM/启动参数优化、Spring 与线程池/连接池配置收敛、依赖与打包裁剪、必要的代码级内存优化 |
| 约束 | 不改变原有功能，可分阶段重启验证，所有变更必须可回滚 |
| 本地验证前提 | 本地启动时允许通过 `com.se.di.tenglong.scada.tool.ApplicationYamlDefaultValueSyncTool#main` 一键替换远程 MQ/数据源等配置 |
| 服务器验证方式 | 将优化后的 jar 传至目标服务器，替换容器 `tenglong-scada-app` 内的 `/app.jar` 后验证 |

### 2.1 验收口径明确化

后续执行时，内存验收统一按以下口径判定，避免“docker stats / top / jcmd”口径不一致：

1. **主验收指标**：容器内 Java 进程 `/proc/<java_pid>/status` 的 `VmRSS`，即 Java 进程实际常驻物理内存。
2. **辅助指标**：宿主机 `docker stats --no-stream tenglong-scada-app` 的 `MEM USAGE`，用于观察容器整体内存，但不作为唯一结论。
3. **观察窗口**：应用进入 `healthy` 后，分别在第 `5 / 15 / 30` 分钟采样；三次 `VmRSS` 均 `< 600 MB`，且无持续上涨趋势，才判定达标。
4. **启动期峰值**：启动过程中允许短时超过 600 MB，但必须记录峰值、持续时间和回落时间；若 30 分钟后仍不回落，则判定未达标。
5. **功能前提**：只有核心接口、MQ 监听、定时任务、数据库访问均通过冒烟后，内存数据才可作为有效验收结果。

统一采样命令如下：

```bash
JAVA_PID=$(docker exec tenglong-scada-app sh -c "pgrep -f 'java.*app.jar' | head -1")
docker exec tenglong-scada-app sh -c "cat /proc/${JAVA_PID}/status | egrep 'VmRSS|VmSize|Threads'"
docker exec tenglong-scada-app sh -c "cat /proc/${JAVA_PID}/smaps_rollup | egrep 'Rss|Pss|Private'"
docker stats --no-stream tenglong-scada-app
```

若 `JAVA_PID` 为空，再使用下面命令人工确认启动命令：

```bash
docker exec tenglong-scada-app sh -c "ps -ef | grep java | grep -v grep"
```

## 3. 非目标

以下内容不在本次任务范围内：

1. 不调整 PostgreSQL、Redis、RabbitMQ 等外部中间件部署形态。
2. 不修改业务流程、接口出入参、数据库结构和持久化语义。
3. 不以牺牲稳定性、吞吐、关键任务可用性为代价换取表面内存下降。
4. 不把账号密码等敏感信息写入仓库文档。
5. 不改变既定部署方式，本次服务器验证沿用“替换容器内 `/app.jar`”模式。

## 4. 总体策略

采用**全链路分阶段型**方案，以“先建立证据，再分层收敛，最后回归验收”为主线，避免盲目调参或误删依赖。

### 4.0 AI 后台执行协议

本任务适合交给 AI 在后台多轮执行，但必须使用**证据驱动闭环**，不得一次性堆叠多个优化项。每轮只允许引入一类主要变更，并按固定流程判断“保留、继续、回滚”。

#### 4.0.1 单轮执行循环

每一轮优化必须按以下顺序执行：

```text
建立本轮假设
  → 记录本轮变更范围
  → 本地修改
  → 本地构建
  → 本地启动或静态验证
  → 上传 jar
  → 替换容器 /app.jar
  → 等待 healthy
  → 采集 5/15/30 分钟内存数据
  → 执行功能冒烟
  → 对比上一轮基线
  → 判定保留 / 继续 / 回滚
```

#### 4.0.2 单轮变更粒度

每轮只允许选择下面一种变更类型：

| 类型 | 示例 | 是否允许和其他类型同轮叠加 |
|---|---|---|
| JVM 参数 | `-Xmx`、`MaxMetaspaceSize`、`MaxDirectMemorySize`、`Xss` | 不允许 |
| 容器/启动环境变量 | `JAVA_OPTS`、Spring profile、日志级别 | 不允许 |
| Spring/Tomcat/Hikari 配置 | Web 线程、连接池、JMX、SpringDoc 开关 | 不允许 |
| 自定义线程池 | `NamedExecutors`、`ThreadPoolTaskExecutor` 参数 | 不允许 |
| 缓存配置 | Caffeine `maximumSize`、TTL、缓存 key 设计 | 不允许 |
| 代码热点 | 删除无界集合、懒加载、避免全量加载 | 不允许 |
| 依赖/自动装配裁剪 | starter、SDK、auto-configuration exclude | 不允许 |

若必须同时调整多个文件才能完成同一项变更，例如一个配置项加默认值并补测试，视为同一轮；但不得把“JVM 参数 + 代码优化 + 依赖裁剪”混在同一轮。

#### 4.0.3 后台执行产物目录

AI 每轮执行时必须在服务器或本地临时目录保存证据，不把敏感信息写入仓库：

```bash
RUN_ID=$(date +%Y%m%d%H%M%S)
RUN_DIR=/tmp/tenglong-scada-memory-runs/${RUN_ID}
mkdir -p "${RUN_DIR}"
```

建议每轮至少保存：

| 文件 | 内容 |
|---|---|
| `hypothesis.md` | 本轮假设、变更点、预期收益、风险 |
| `before.txt` | 替换前 `docker stats`、`VmRSS`、线程数、启动参数 |
| `after-5m.txt` | healthy 后第 5 分钟采样 |
| `after-15m.txt` | healthy 后第 15 分钟采样 |
| `after-30m.txt` | healthy 后第 30 分钟采样 |
| `smoke.txt` | 冒烟命令与结果 |
| `decision.md` | 保留/继续/回滚结论和原因 |
| `rollback.txt` | 如发生回滚，记录回滚命令和回滚后 sha256 |

#### 4.0.4 判定门禁

每轮结束必须按下表判定：

| 判定 | 条件 | 后续动作 |
|---|---|---|
| 保留 | 30 分钟内 `VmRSS < 600 MB`，功能冒烟通过，日志无新关键异常 | 固化变更，进入最终验收 |
| 继续 | 功能正常，RSS 有下降但仍未达标，或证据显示还有明确热点 | 保留本轮变更，进入下一轮 |
| 回滚 | 容器无法 healthy、核心链路失败、RSS 明显升高、GC 频繁 Full GC、线程/连接明显堆积 | 立即回滚旧 jar 或旧参数 |
| 暂停 | 证据无法解释、连续 3 轮无收益、收益来自牺牲功能 | 停止自动优化，输出人工决策问题 |

#### 4.0.5 连续迭代停止条件

AI 后台执行必须在以下任一条件满足时停止：

1. 达成 `healthy` 后 5/15/30 分钟 `VmRSS < 600 MB`。
2. 连续 3 轮优化后 RSS 下降均 `< 5%`。
3. 出现功能异常并且回滚后恢复。
4. 需要变更业务语义、接口契约、数据库结构或部署拓扑。
5. 需要删除不确定用途的依赖或禁用不确定用途的业务组件。
6. 目标机器、账号、容器名、启动方式与文档不一致。

#### 4.0.6 每轮优化记录模板

每轮开始前复制以下模板到 `hypothesis.md`：

```markdown
# Memory Optimization Run <RUN_ID>

## Hypothesis

- Problem source:
- Evidence:
- Planned change:
- Expected RSS impact:
- Functional risk:
- Rollback method:

## Change Scope

- Files changed:
- Runtime parameters changed:
- Dependencies changed:

## Verification Plan

- Build command:
- Local verification:
- Remote deployment:
- Memory sampling:
- Smoke checks:

## Decision

- Result: keep / continue / rollback / pause
- Reason:
- Next step:
```

### 阶段一：基线采集

目标：建立当前内存占用画像，明确内存主要消耗在堆、元空间、直接内存、线程栈、缓存还是第三方组件。

需要固化的证据包括：

1. 当前启动命令、JVM 参数、环境变量。
2. 进程 RSS/RES、VIRT、线程数、文件句柄、容器/宿主机限制。
3. 堆使用情况、GC 行为、类加载数量、线程栈数量。
4. 依赖树、已启用的 Spring Bean、定时任务、消息监听器、缓存组件、文档组件、监控组件。
5. 关键运行场景下的内存快照证据，如 `jcmd`、`jmap`、`jstat`、`JFR`、`GC.class_histogram`。

#### 阶段一输出格式

基线采集完成后必须输出以下表格，作为后续所有优化的对照基线：

| 指标 | 采集命令 | 当前值 | 说明 |
|---|---|---|---|
| 容器状态 | `docker ps --filter name=tenglong-scada-app` |  | 是否 healthy |
| 容器内 Java PID | `pgrep -f 'java.*app.jar'` |  | 后续命令使用该 PID |
| VmRSS | `/proc/<pid>/status` |  | 主验收指标 |
| VmSize | `/proc/<pid>/status` |  | 虚拟内存参考 |
| Threads | `/proc/<pid>/status` |  | 线程栈内存判断 |
| Heap used / committed | `jcmd <pid> GC.heap_info` |  | 判断堆是否主因 |
| Metaspace | `jcmd <pid> VM.metaspace` 或 NMT |  | 判断类元数据占用 |
| Class count | `jcmd <pid> GC.class_stats` 或 histogram |  | 依赖裁剪参考 |
| Direct memory | NMT 或启动参数 |  | 判断堆外内存 |
| Hikari pool size | 日志/配置/env |  | 连接池内存与连接数 |
| Java flags | `jcmd <pid> VM.flags` |  | 参数基线 |
| Top heap classes | `jcmd <pid> GC.class_histogram` |  | 代码热点候选 |

如果某个命令不可用，必须在“当前值”中写明“不可用”及原因，不允许空着。

#### JVM 诊断前置条件

`jcmd VM.native_memory summary` 依赖 JVM 启动参数，默认不一定可用。执行时必须按下面规则处理：

1. **普通基线轮**：不强制开启 NMT，仅采集 `VmRSS / smaps_rollup / heap_info / class_histogram / Thread.print`。
2. **NMT 诊断轮**：若 RSS 来源不清楚，允许临时增加以下启动参数并重启一次，只用于诊断：
   ```bash
   -XX:NativeMemoryTracking=summary
   ```
3. NMT 诊断轮必须单独记录，因为开启 NMT 本身会带来少量额外开销，不可直接与普通轮 RSS 做绝对值对比。
4. 若容器内没有 `jcmd/jstat`，不得因此阻塞任务，应使用 `/proc/<pid>/status`、`/proc/<pid>/smaps_rollup`、`docker stats` 继续完成主验收。

### 阶段一补充：本地验证准备

在进入服务器替换前，先在本地完成一次可运行验证：

1. 使用 `com.se.di.tenglong.scada.tool.ApplicationYamlDefaultValueSyncTool#main` 同步远程 MQ、数据源等默认配置。
2. 在尽量贴近目标环境的前提下启动本地应用，确认核心链路可运行。
3. 本地先验证启动参数、依赖裁剪、代码修正不会导致显性启动失败。
4. 仅当本地验证通过后，再进入服务器容器内 jar 替换验证。

#### 本地执行步骤（Windows PowerShell）

```powershell
# 1. 编译测试类，确保 ApplicationYamlDefaultValueSyncTool 可运行
Set-Location C:\Project\tenglong-scada
mvn -pl tenglong-scada-webapi -am test-compile -DskipTests

# 2. 通过 IDE 直接运行以下 main（推荐）
# com.se.di.tenglong.scada.tool.ApplicationYamlDefaultValueSyncTool#main

# 3. 若需命令行运行，先导出测试运行 classpath
mvn -pl tenglong-scada-webapi -DskipTests dependency:build-classpath "-Dmdep.outputFile=target\test-classpath.txt" "-Dmdep.pathSeparator=;"

# 4. 拼出 classpath 并执行同步工具
$cp = "tenglong-scada-webapi\target\test-classes;tenglong-scada-webapi\target\classes;" + (Get-Content "tenglong-scada-webapi\target\test-classpath.txt" -Raw)
java -cp $cp com.se.di.tenglong.scada.tool.ApplicationYamlDefaultValueSyncTool --container-name tenglong-scada-app

# 5. 构建本地可运行 jar
mvn clean install -P development

# 6. 确认产物位置，并选取本轮要验证的 jar
$jar = Get-ChildItem .\tenglong-scada-webapi\target\*.jar |
    Where-Object { $_.Name -notlike '*.original' } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
$jar.FullName
```

#### 本地启动与内存观察步骤（Windows PowerShell）

```powershell
Set-Location C:\Project\tenglong-scada

# 1. 指定本轮要启动的 jar（沿用上一步得到的 $jar）
$jar = Get-ChildItem .\tenglong-scada-webapi\target\*.jar |
    Where-Object { $_.Name -notlike '*.original' } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

# 2. 启动应用（示例参数，后续按优化方案调整）
& java `
  '-Xms256m' `
  '-Xmx256m' `
  '-XX:MaxMetaspaceSize=128m' `
  '-XX:MaxDirectMemorySize=128m' `
  '-XX:ReservedCodeCacheSize=96m' `
  '-jar' $jar.FullName

# 3. 另开一个终端查看 Java 进程
Get-Process java | Sort-Object WS -Descending | Select-Object -First 5 Id,ProcessName,WS,PM,VM,Path

# 4. 若安装了 JDK 工具，补充查看堆/原生内存
jcmd <PID> VM.flags
jcmd <PID> GC.heap_info
jcmd <PID> VM.native_memory summary
jcmd <PID> Thread.print
```

本地验证至少要确认以下内容：

1. 应用能正常启动，无明显 Bean 装配缺失。
2. 关键依赖（MQ、数据库、缓存）可正常连接。
3. 核心接口、定时任务、监听器不出现启动期异常。
4. 每轮参数或代码调整后都能复现启动并记录内存表现。

#### 本地启动与内存观察步骤（macOS / Linux）

当前开发环境如为 macOS 或 Linux，按下面命令执行本地验证：

```bash
cd /Users/puremdq/Documents/Project/tenglong-scada

# 1. 构建本地 jar
mvn clean install -P development

# 2. 选择最新可运行 jar
JAR=$(find tenglong-scada-webapi/target -maxdepth 1 -name '*.jar' ! -name '*.original' -print | sort | tail -1)
echo "$JAR"

# 3. 启动应用；参数按每轮优化方案调整
java \
  -Xms256m \
  -Xmx256m \
  -XX:MaxMetaspaceSize=128m \
  -XX:MaxDirectMemorySize=128m \
  -XX:ReservedCodeCacheSize=96m \
  -jar "$JAR"

# 4. 另开终端观察 Java 进程
PID=$(pgrep -f 'java.*tenglong-scada' | head -1)
ps -o pid,rss,vsz,nlwp,command -p "$PID"
jcmd "$PID" VM.flags
jcmd "$PID" GC.heap_info
jcmd "$PID" Thread.print | head -200
```

若本地无法连接远程中间件，本地验证只作为“启动与显性装配问题”检查，最终 RSS 验收仍以目标服务器为准。

### 阶段二：参数与运行时配置优化

目标：在不改代码的前提下先消除明显的配置性浪费。

重点检查项：

1. `-Xms/-Xmx` 是否远大于实际需要。
2. `MaxMetaspaceSize`、`ReservedCodeCacheSize`、`MaxDirectMemorySize` 是否缺少上限。
3. GC 策略是否适配当前机器资源与负载。
4. Tomcat 线程数、队列、连接数是否显著偏大。
5. Hikari 连接池配置是否超出真实并发需求。
6. 自定义线程池、定时线程池、重试线程池是否存在过配。
7. SpringDoc、JMX、调试日志、Banner、开发期开关是否在生产环境不必要开启。

#### 阶段二执行优先级

参数与配置优化按下面顺序执行，上一项未验证前不得叠加下一项：

1. **JVM 内存上限**：明确 `-Xms/-Xmx`、`MaxMetaspaceSize`、`MaxDirectMemorySize`、`ReservedCodeCacheSize`。
2. **线程栈与线程数**：确认 `-Xss`，统计 Java 线程数，优先减少长期空闲线程池。
3. **Tomcat 配置**：收敛 `maxThreads`、`acceptCount`、连接超时等 Web 容器参数。
4. **Hikari 连接池**：按真实并发收敛 config/runtime 两个数据源的 `maximumPoolSize/minimumIdle`。
5. **自定义 Executor**：检查所有 `NamedExecutors`、`ThreadPoolTaskExecutor`、静态线程池。
6. **Caffeine 缓存**：检查最大容量、TTL、是否存在按项目/点位无限增长的 key。
7. **非必要运行特性**：关闭生产不需要的 SpringDoc、JMX、调试日志、开发期开关。

#### 阶段二推荐首轮参数候选

首轮参数优化必须基于基线数据决定，不能直接照抄。若基线显示 RSS 主要来自 JVM 预留、堆提交、线程栈或元空间，可从以下候选中一次只选一个方向：

| 候选 | 适用证据 | 示例 | 风险 |
|---|---|---|---|
| 降低初始堆 | `Xms` 明显大于稳定堆使用 | `-Xms128m` 或 `-Xms256m` | 启动期 GC 增加 |
| 限制最大堆 | 稳定 heap used 远低于 Xmx | `-Xmx256m` / `-Xmx384m` | 高峰期 OOM |
| 限制元空间 | 类加载稳定且 Metaspace 低 | `-XX:MaxMetaspaceSize=192m` | 类加载失败 |
| 限制直接内存 | DirectMemory 不大但未设上限 | `-XX:MaxDirectMemorySize=96m` | 网络/IO buffer 不足 |
| 降低线程栈 | 线程数较多且栈默认偏大 | `-Xss512k` | 深调用栈 StackOverflow |
| 开启容器感知比例 | 容器 limit 明确且 JVM 未按容器限制 | `-XX:MaxRAMPercentage=...` | 与显式 Xmx 同时配置时需明确优先级 |

每次参数调整后，必须同时记录：启动耗时、GC 次数、Full GC 次数、接口冒烟耗时、RSS 变化。

#### 远端 JVM 启动参数应用方式

远端 `tenglong-scada-app` 的 JVM 启动参数不直接在容器内修改，也不通过临时 `docker run` 参数修改。标准方式是修改宿主机环境文件中的 `JAVA_OPTS_XM_TENGLONG_SCADA_APP`，再通过既有 `docker-compose` 文件重新应用。

固定文件路径如下：

| 项目 | 路径 |
|---|---|
| 环境变量文件 | `/opt/eeo/application/docker-compose/env/scada.env` |
| compose 文件 | `/opt/eeo/application/docker-compose/runtime/tenglong-scada-app-1.5.0.yml` |
| JVM 参数变量 | `JAVA_OPTS_XM_TENGLONG_SCADA_APP` |

标准操作命令：

```bash
ssh root@10.235.236.25

# 1. 备份 env 文件
RUN_ID=$(date +%Y%m%d%H%M%S)
cp /opt/eeo/application/docker-compose/env/scada.env \
  /tmp/scada.env.bak-${RUN_ID}

# 2. 查看当前 JVM 参数
grep '^JAVA_OPTS_XM_TENGLONG_SCADA_APP=' \
  /opt/eeo/application/docker-compose/env/scada.env

# 3. 修改 JAVA_OPTS_XM_TENGLONG_SCADA_APP
# 可用 vi 手工修改，或使用明确的 sed 命令；禁止把密码等敏感 env 输出到仓库文档。
vi /opt/eeo/application/docker-compose/env/scada.env

# 4. 应用 compose 配置
docker-compose \
  -f "/opt/eeo/application/docker-compose/runtime/tenglong-scada-app-1.5.0.yml" \
  --env-file "/opt/eeo/application/docker-compose/env/scada.env" \
  up -d
```

##### compose 应用后 jar 保持性检查

必须注意：当前部署验证方式是手工 `docker cp` 替换容器内 `/app.jar`。如果 `docker-compose up -d` 因环境变量变化而**重建容器**，新容器会从镜像重新创建，容器文件系统会恢复到镜像初始状态，之前手工替换过的 `/app.jar` 很可能丢失。因此，启动参数变更轮必须把 compose 操作视为“可能回滚 jar”的高风险动作。

每次执行 `docker-compose up -d` 前后必须记录容器 ID 和 jar sha256：

```bash
# compose 前记录容器与当前 jar
CONTAINER_ID_BEFORE=$(docker inspect tenglong-scada-app --format '{{.Id}}')
docker cp tenglong-scada-app:/app.jar /tmp/app.jar.before-compose
sha256sum /tmp/app.jar.before-compose
sha256sum /tmp/app-optimized.jar

# 应用 compose
docker-compose \
  -f "/opt/eeo/application/docker-compose/runtime/tenglong-scada-app-1.5.0.yml" \
  --env-file "/opt/eeo/application/docker-compose/env/scada.env" \
  up -d

# compose 后记录容器与当前 jar
CONTAINER_ID_AFTER=$(docker inspect tenglong-scada-app --format '{{.Id}}')
docker cp tenglong-scada-app:/app.jar /tmp/app.jar.after-compose
sha256sum /tmp/app.jar.after-compose
echo "before=${CONTAINER_ID_BEFORE}"
echo "after=${CONTAINER_ID_AFTER}"
```

判定规则：

1. 若 `CONTAINER_ID_BEFORE != CONTAINER_ID_AFTER`，说明容器已重建，必须检查 `/app.jar` 是否仍为本轮优化 jar。
2. 若 `/tmp/app.jar.after-compose` 的 sha256 与 `/tmp/app-optimized.jar` 不一致，说明 jar 已恢复为镜像内初始 jar 或其他版本，必须再次替换 `/app.jar`。
3. 即使容器 ID 未变化，也必须做 sha256 对比，不能只凭容器状态判断。
4. 替换 jar 后必须再次启动容器并确认 JVM 参数已生效。

二次替换流程：

```bash
# 仅当 app.jar sha256 与本轮优化 jar 不一致时执行
docker stop tenglong-scada-app
docker cp /tmp/app-optimized.jar tenglong-scada-app:/app.jar
docker start tenglong-scada-app

# 确认 jar 与启动参数都正确
docker cp tenglong-scada-app:/app.jar /tmp/app.jar.after-recopy
sha256sum /tmp/app.jar.after-recopy
sha256sum /tmp/app-optimized.jar

JAVA_PID=$(docker exec tenglong-scada-app sh -c "pgrep -f 'java.*app.jar' | head -1")
docker exec tenglong-scada-app sh -c "tr '\0' ' ' < /proc/${JAVA_PID}/cmdline"
docker exec tenglong-scada-app jcmd "$JAVA_PID" VM.flags
```

启动参数变更轮的有效性必须同时满足：

1. `JAVA_OPTS_XM_TENGLONG_SCADA_APP` 已在 `scada.env` 中更新。
2. `docker-compose up -d` 已执行成功。
3. 容器内 `/app.jar` sha256 与本轮优化 jar 一致。
4. Java 进程实际启动命令或 `jcmd VM.flags` 能看到本轮 JVM 参数。
5. 容器恢复 healthy 并通过最小功能冒烟。

### 阶段三：依赖与打包裁剪

目标：识别并去除不参与当前运行路径的包、starter、SDK 和附属能力，减少类加载、Bean 装配、线程创建和缓存初始化。

裁剪原则：

1. 先做“代码引用可达性”检查，再做“启动路径”检查，最后做“运行路径”检查。
2. 对可选能力优先采用关闭自动装配、关闭配置开关、延迟初始化，而不是直接粗暴删除。
3. 对消息、文档、外部 SDK、图形处理、监控、测试残留依赖进行重点排查。
4. 所有裁剪项必须有对应的功能验证清单与回滚方式。

#### 阶段三准入门槛

依赖与自动装配裁剪风险较高，只有满足以下条件之一才允许进入：

1. 基线显示 Metaspace、类数量、Spring Bean 数量是主要内存来源。
2. `GC.class_histogram` 或启动日志显示某个 SDK/文档/监控组件加载大量类或对象。
3. 该依赖只用于测试、开发、文档或未启用功能，并且有明确引用检查证据。

裁剪顺序必须是：

```text
配置关闭 / exclude auto-configuration
  → 延迟初始化 / 条件装配
  → 缩小扫描范围
  → 最后才允许删除 pom 依赖
```

删除 pom 依赖前必须完成：

```bash
mvn -q -DskipTests dependency:tree > /tmp/dependency-tree.before.txt
rg "目标依赖关键包名|目标依赖类名" .
mvn test-compile -DskipTests
```

若引用关系不明确，禁止删除依赖。

### 阶段四：代码级内存热点治理

目标：针对真正占用内存的热点做最小必要改动，不做无关重构。

优先排查方向：

1. 长生命周期缓存、静态 Map、重复缓存、无上限集合。
2. AOP、反射、动态代理、翻译/校验/过滤等横切逻辑带来的对象滞留。
3. 定时任务、消息监听器、Runner、重试框架带来的线程和上下文常驻。
4. 大对象构造、批量装配、一次性全量加载、重复 DTO/Model/PO 转换。
5. 不必要的 Bean 初始化、文档扫描、配置扫描、外部客户端预热。
6. Direct buffer、序列化缓冲区、线程本地变量、TTL/上下文传播对象。

#### 阶段四优先排查代码点

结合当前项目代码结构，代码级热点优先从以下位置开始排查：

| 优先级 | 代码点 | 排查原因 | 期望处理方式 |
|---|---|---|---|
| P0 | `UnifiedAopAspect` / ShadowMatch 相关逻辑 | AOP 候选方法多时容易产生大量反射/AOP 缓存对象 | 保留业务语义，减少不必要切点匹配与缓存膨胀 |
| P0 | `NamedExecutors` 及其调用点 | 默认按 CPU 核数创建线程池，长期线程会占用栈内存 | 对长期运行线程池设置明确上限与命名，避免重复创建 |
| P0 | `EventServiceImpl` 的 `event-svc` executor | 静态 IO 线程池常驻 | 根据实际并发收敛线程数，确认无任务堆积 |
| P0 | `DriverTagActiveGatewayImpl#listAllTagInfoMap` | 全量 driver tag 缓存可能随数据量膨胀 | 评估缓存容量、TTL、按项目隔离和刷新时机 |
| P0 | `VirtualTagValueRefreshTask` | 定时刷新可能触发批量查询、缓存装配和线程常驻 | 控制调度频率、批量大小、并发度 |
| P1 | `LocalCacheConfig` 中各 Caffeine cache | 部分缓存容量较大或无过期 | 为每个缓存明确最大容量与 TTL，不影响业务命中语义 |
| P1 | MQ Listener / Runner / 定时任务 | 启动后自动创建线程与上下文 | 明确哪些任务在当前部署必须启用，非必须任务用配置开关控制 |
| P2 | SpringDoc / 外部 SDK / 文档扫描 | 类加载和 Bean 初始化成本高 | 优先关闭自动装配或延迟初始化，不直接删依赖 |

#### 阶段四代码修改规则

代码级优化必须遵守以下规则：

1. 优先消除无界增长、重复缓存、重复线程池、全量加载；不做大规模重构。
2. 不改变接口返回结构、数据库查询语义、MQ 消息格式、定时任务业务语义。
3. 每个代码优化必须补一个最小测试或最小复现验证；无法自动化时必须给出命令级验证证据。
4. 对缓存优化必须说明 key、容量、TTL、失效时机和工程切换行为。
5. 对线程池优化必须说明核心线程数、最大线程数、队列长度、拒绝策略和任务堆积观察方式。
6. 对全量加载优化必须说明数据量级、批大小、是否分页、是否会改变排序或过滤。

#### 阶段四候选 Backlog

后续 AI 可按下面 Backlog 逐项调查，但不能无证据直接修改：

| 编号 | 候选项 | 调查命令/证据 | 可接受优化方向 |
|---|---|---|---|
| B1 | AOP 切点缓存膨胀 | `jcmd <pid> GC.class_histogram \| grep -E 'ShadowMatch\|AspectJ'` | 缩小切点、去掉无效 AOP、减少候选方法匹配 |
| B2 | 静态线程池过多 | `jcmd <pid> Thread.print`、`grep NamedExecutors` | 合并线程池、降低线程数、增加关闭钩子 |
| B3 | DriverTag 全量缓存 | histogram 中 `DriverTag` / Map 占比、数据行数 | 限容量、按需加载、按项目清理 |
| B4 | 虚拟点刷新批量装配 | 线程栈、日志、SQL、刷新周期 | 降低频率、分页处理、缓存复用 |
| B5 | Caffeine cache 过大 | cache 配置、key 数、对象直方图 | 设置最大容量、TTL、工程切换清理 |
| B6 | SpringDoc/文档组件 | 类直方图、Bean 列表 | 生产关闭或条件装配 |
| B7 | 外部 SDK 常驻对象 | 类直方图、Bean 列表 | 懒初始化、按需创建 |
| B8 | 日志异步队列 | 线程和队列对象直方图 | 调整队列容量、日志级别 |

### 阶段五：回归与验收

目标：证明优化结果真实、稳定、可回滚。

#### 服务器容器替换验证命令

以下命令作为后续执行阶段的标准替换流程，默认新包先上传到服务器临时目录，再替换容器 `tenglong-scada-app` 内的 `/app.jar`：

```bash
# 0. 上传新包到服务器临时目录（在本地执行）
scp ./tenglong-scada-webapi/target/<latest-built-jar>.jar root@10.235.236.25:/tmp/app-optimized.jar

# 1. 登录服务器
ssh root@10.235.236.25

# 2. 确认目标容器存在
docker ps -a --filter "name=tenglong-scada-app"

# 3. 备份当前容器内 jar 到宿主机
docker cp tenglong-scada-app:/app.jar /tmp/app.jar.bak

# 4. 停止容器
docker stop tenglong-scada-app

# 5. 将新 jar 拷贝进容器固定位置
docker cp /tmp/app-optimized.jar tenglong-scada-app:/app.jar

# 6. 启动容器
docker start tenglong-scada-app

# 7. 观察容器状态与应用日志
docker ps --filter "name=tenglong-scada-app"
docker logs --tail 200 tenglong-scada-app

# 8. 获取容器内 Java 进程 PID
docker exec tenglong-scada-app sh -c "ps -ef | grep java | grep -v grep"

# 9. 在宿主机观察容器资源
docker stats --no-stream tenglong-scada-app

# 10. 若容器内存在 jcmd，再补充 JVM 侧证据
JAVA_PID=$(docker exec tenglong-scada-app sh -c "pgrep -f 'java.*app.jar' | head -1")
docker exec tenglong-scada-app jcmd "$JAVA_PID" VM.flags
docker exec tenglong-scada-app jcmd "$JAVA_PID" GC.heap_info
docker exec tenglong-scada-app jcmd "$JAVA_PID" VM.native_memory summary
```

如需回滚，直接使用已备份的 jar 反向执行一次：

```bash
docker stop tenglong-scada-app
docker cp /tmp/app.jar.bak tenglong-scada-app:/app.jar
docker start tenglong-scada-app
docker logs --tail 200 tenglong-scada-app
```

#### 服务器侧基线采集命令

在正式替换前，应先在服务器执行一次基线采集，保留当前版本的内存证据：

```bash
ssh root@10.235.236.25

# 1. 容器与资源概览
docker ps -a --filter "name=tenglong-scada-app"
docker stats --no-stream tenglong-scada-app

# 2. 容器内 Java 进程信息
docker exec tenglong-scada-app sh -c "ps -ef | grep java | grep -v grep"
JAVA_PID=$(docker exec tenglong-scada-app sh -c "pgrep -f 'java.*app.jar' | head -1")
docker exec tenglong-scada-app sh -c "cat /proc/${JAVA_PID}/status | egrep 'VmRSS|VmSize|Threads'"
docker exec tenglong-scada-app sh -c "cat /proc/${JAVA_PID}/smaps_rollup | egrep 'Rss|Pss|Private'"

# 3. JVM 证据（容器内具备 jcmd/jstat 时）
docker exec tenglong-scada-app jcmd "$JAVA_PID" VM.flags
docker exec tenglong-scada-app jcmd "$JAVA_PID" GC.heap_info
docker exec tenglong-scada-app jcmd "$JAVA_PID" VM.native_memory summary
docker exec tenglong-scada-app jcmd "$JAVA_PID" GC.class_histogram
docker exec tenglong-scada-app jstat -gcutil "$JAVA_PID" 1000 10

# 4. 启动参数与环境变量
docker exec tenglong-scada-app sh -c "tr '\0' ' ' < /proc/1/cmdline"
docker inspect tenglong-scada-app --format '{{json .Config.Env}}'
```

#### 每一轮优化后的固定验证动作

每做完一轮参数、依赖或代码优化，都按下面顺序执行：

1. 本地重新构建 jar。
2. 本地启动并完成一次核心链路验证。
3. 记录本地 Java 进程内存、线程、GC 指标。
4. 上传 jar 到服务器临时目录。
5. 若本轮涉及 JVM 启动参数，先修改 `/opt/eeo/application/docker-compose/env/scada.env` 中的 `JAVA_OPTS_XM_TENGLONG_SCADA_APP`，再执行既定 `docker-compose up -d`。
6. compose 应用后必须校验容器内 `/app.jar` sha256；若被恢复为镜像初始 jar，立即二次替换 `/app.jar`。
7. 按容器替换流程更新或确认 `/app.jar`。
8. 采集 `docker stats`、`VmRSS`、`VM.native_memory summary`。
9. 对核心接口、MQ、定时任务做最小冒烟。
10. 若内存未达标，保留证据进入下一轮；若功能异常，立即回滚。

#### 最小功能冒烟清单

每轮内存数据只有在最小功能冒烟通过后才有效。冒烟清单按“尽量少但覆盖关键链路”的原则执行：

| 链路 | 检查方式 | 通过标准 |
|---|---|---|
| 容器健康 | `docker ps --filter "name=tenglong-scada-app"` | 状态为 `healthy` 或应用日志显示启动完成 |
| 启动日志 | `docker logs --tail 300 tenglong-scada-app` | 无新的 `ERROR`、数据库连接失败、Bean 装配失败 |
| 数据库连接 | 观察 Hikari 启动日志、连接池状态日志、核心接口访问 | config/runtime 数据源无连接失败 |
| MQ 监听 | 观察 RabbitMQ listener 启动日志 | 无队列声明失败、认证失败、消费线程异常退出 |
| 定时任务 | 观察 5 到 15 分钟日志 | 关键 scheduler 无连续异常 |
| REST 接口 | 使用现场已有健康接口或核心查询接口 | 返回成功响应，耗时无明显恶化 |
| 连接数 | 宿主机 `ss -antp` 或数据库侧连接视图 | 连接数不出现持续上涨或打满 |

若现场没有统一健康接口，则以 Docker health、启动日志、数据库/MQ 连接日志和一个已知核心业务接口作为替代。AI 不得自行新增健康接口作为本任务的一部分。

#### 内存采样记录格式

每次采样都按下面格式记录，便于后续 AI 横向比较：

| 时间点 | VmRSS | VmSize | Threads | Heap used | Heap committed | Docker MEM | Full GC | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| before |  |  |  |  |  |  |  |  |
| 5m |  |  |  |  |  |  |  |  |
| 15m |  |  |  |  |  |  |  |  |
| 30m |  |  |  |  |  |  |  |  |

判定时优先看 `VmRSS` 和趋势；如果 heap 很低但 RSS 高，下一轮优先调查线程栈、DirectMemory、Metaspace、JNI/native、mmap 文件和第三方客户端。

#### 回滚证据要求

每次替换前必须记录以下信息，保证可回滚且能证明回滚成功：

```bash
docker inspect tenglong-scada-app --format '{{.Image}}'
docker inspect tenglong-scada-app --format '{{json .Config.Env}}' > /tmp/tenglong-scada-app.env.before.json
docker inspect tenglong-scada-app --format '{{json .HostConfig}}' > /tmp/tenglong-scada-app.hostconfig.before.json
docker cp tenglong-scada-app:/app.jar /tmp/app.jar.bak
sha256sum /tmp/app.jar.bak
sha256sum /tmp/app-optimized.jar
```

回滚后必须再次执行：

```bash
docker start tenglong-scada-app
docker ps --filter "name=tenglong-scada-app"
docker logs --tail 200 tenglong-scada-app
docker cp tenglong-scada-app:/app.jar /tmp/app.jar.after-rollback
sha256sum /tmp/app.jar.after-rollback
```

验收要求：

1. 本地验证通过，且与远程 MQ/数据源等关键依赖联通正常。
2. 服务器侧按既定方式替换容器内 `/app.jar` 后应用启动成功。
3. 核心业务接口、关键定时任务、消息监听、缓存、数据访问链路可用。
4. Java 进程 RSS/RES 在稳定运行窗口内连续观察低于 `600 MB`。
5. 所有变更项具备“变更内容、收益、风险、回滚方式”记录。

## 5. AI 执行时的明确输入

为了让后续执行 AI 更聚焦，任务输入应固定为下面的约束：

### 5.1 必须满足

1. 目标应用：`10.235.236.25` 上的 `tenglong-scada-app`。
2. 唯一核心指标：Java 进程 RSS/RES 常驻内存 `< 600 MB`。
3. 不允许改变现有业务功能和对外行为。
4. 允许使用参数优化、依赖裁剪、代码优化三类手段。
5. 允许分阶段重启验证，但每一步都必须可以回滚。
6. 执行顺序默认包含“本地验证 → 服务器容器替换验证”。

### 5.2 必须输出

1. 当前内存基线与证据。
2. 内存问题来源分解：堆、元空间、直接内存、线程、缓存、第三方组件。
3. 每一项优化措施的原因、改动点、预期收益、验证方式。
4. 功能等价性验证清单。
5. 最终达标结论或未达标原因。

### 5.3 明确禁止

1. 未验证即删除依赖或禁用组件。
2. 通过降低关键功能可用性来换取内存下降。
3. 只看堆内存、不看 RSS/RES 总占用。
4. 不保留回滚手段的直接覆盖式修改。

### 5.4 推荐给后台 AI 的任务输入

后续如果把任务交给后台 AI，可直接使用下面输入，避免目标和边界丢失：

```text
请按 docs/2026-07-10-tenglong-scada-app-memory-optimization-design.md 执行 tenglong-scada-app 内存优化。

硬目标：
1. 目标容器 tenglong-scada-app，主验收指标为容器内 Java 进程 VmRSS。
2. 应用 healthy 后第 5/15/30 分钟 VmRSS 均需 < 600 MB。
3. 不允许改变业务功能、接口契约、数据库语义、MQ 消息语义和部署拓扑。
4. 每轮只允许一种主要变更，必须保留证据和回滚方式。

执行流程：
1. 先采集当前基线，不要先改代码。
2. 建立本轮假设，写明证据、预期收益和回滚方法。
3. 本地修改、构建、启动或静态验证。
4. 上传 jar，替换容器 /app.jar，等待 healthy。
5. 采集 5/15/30 分钟 VmRSS、docker stats、heap、线程、GC。
6. 做最小功能冒烟。
7. 根据门禁判定保留、继续、回滚或暂停。
8. 若连续 3 轮收益不足 5%，或需要业务/数据库/部署决策，停止并输出问题。

禁止：
1. 不得把密码、token、连接串明文写入仓库。
2. 不得无证据删除依赖。
3. 不得一次混合 JVM 参数、配置、代码、依赖多类变更。
4. 不得用牺牲功能可用性的方式换内存下降。
```

### 5.5 后台 AI 最终输出格式

后台 AI 完成或暂停时，必须输出以下结论：

```markdown
## Final Result

- Status: reached / not reached / rolled back / paused
- Final VmRSS 5m / 15m / 30m:
- Baseline VmRSS:
- RSS delta:
- Final jar sha256:
- Rollback jar sha256:

## Changes Kept

| Round | Change | Evidence | RSS delta | Risk |
|---|---|---|---:|---|

## Changes Reverted

| Round | Change | Revert reason |
|---|---|---|

## Validation

- Build:
- Local startup:
- Remote container:
- Smoke checks:
- Logs:

## Remaining Risks

-

## Next Human Decision Needed

-
```

## 6. 推荐执行顺序

1. 先在服务器侧拿到真实基线数据。
2. 记录旧 jar sha256、容器镜像、启动参数、环境变量，准备回滚包。
3. 先做低风险 JVM 参数和配置收敛。
4. 再收敛 Tomcat、Hikari、自定义线程池和缓存上限。
5. 只有在证据证明依赖/自动装配是主要来源时，才进入依赖裁剪。
6. 最后做代码级热点优化。
7. 本地验证通过后，再上传 jar 并替换服务器容器内 `/app.jar`。
8. 每一轮都执行功能回归和 RSS/RES 复测。
9. 每轮只允许引入一类主要变更，避免无法判断收益来源。

## 7. 风险与控制

| 风险 | 说明 | 控制措施 |
|---|---|---|
| 误删依赖 | 编译能过但运行路径报错 | 必须做引用可达性、启动路径、运行路径三级校验 |
| 参数过度压缩 | 内存下降但频繁 GC 或吞吐恶化 | 每次调参同时观察 GC、响应时间、线程状态 |
| 线程池过度收敛 | 内存下降但任务堆积 | 结合实际并发与任务模型逐项压测/观察 |
| 缓存误优化 | 命中率下降或功能异常 | 只清理无上限或重复缓存，保留业务缓存语义 |
| 隐性堆外内存 | 堆看似正常但 RSS 不降 | 强制纳入 DirectMemory、线程栈、JNI、类元数据分析 |

## 8. 文档产出物

后续正式执行时，建议至少输出以下内容：

1. 服务器现状基线报告。
2. JVM/配置优化清单。
3. 依赖裁剪评估清单。
4. 代码热点修复清单。
5. 本地验证记录。
6. 服务器替换与验收回滚报告。

## 9. 结论

这不是单纯“调小堆”任务，而是一个以 **RSS/RES < 600 MB** 为硬指标的分阶段内存治理任务。最合理的推进方式是：

**基线采集 → 参数优化 → 依赖裁剪 → 代码热点治理 → 回归验收**。

后续 AI 执行时，应严格围绕该顺序推进，确保每一步都有证据、有边界、有回滚，而不是直接做高风险改动。
