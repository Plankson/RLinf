#!/bin/bash
set -e

# ================= 配置区 =================
# 定义关键路径
LIB_DIR="/usr/lib/x86_64-linux-gnu"
NVIDIA_URL_BASE="https://us.download.nvidia.com/tesla"
# =========================================

echo ">>> [Auto-Fix-EGL] 开始检查图形渲染环境..."

# 1. 获取宿主机驱动版本 (例如: 570.172.08)
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: 找不到 nvidia-smi，无法检测驱动版本。"
    exit 1
fi
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d ' ')
echo ">>> 检测到宿主机驱动版本: ${DRIVER_VERSION}"

# 2. 清理版本不匹配的残留库 (解决“脏镜像”问题)
# 如果容器里有 550 的库，但现在跑在 570 上，必须删掉旧的
echo ">>> 清理版本不匹配的旧库..."
# find ${LIB_DIR} -name "libnvidia-*.so.*" ! -name "*${DRIVER_VERSION}*" -delete
# find ${LIB_DIR} -name "libcuda.so.*" ! -name "*${DRIVER_VERSION}*" -delete

# 3. 检查关键文件是否缺失
# Driver 570+ 强依赖 libnvidia-gpucomp.so，旧版 Toolkit 常漏挂载此文件
# MISSING_FILE=false
# if [ ! -f "${LIB_DIR}/libnvidia-gpucomp.so.${DRIVER_VERSION}" ]; then
#     echo "!!! 发现缺失关键库: libnvidia-gpucomp.so.${DRIVER_VERSION}"
#     MISSING_FILE=true
# fi

# # 检查 glcore (通常不会缺，但为了保险起见一并检查)
# if [ ! -f "${LIB_DIR}/libnvidia-glcore.so.${DRIVER_VERSION}" ]; then
#     echo "!!! 发现缺失关键库: libnvidia-glcore.so.${DRIVER_VERSION}"
#     MISSING_FILE=true
# fi
# MISSING_FILE=true
# # 4. 如果缺失，下载并提取
# if [ "$MISSING_FILE" = true ]; then
echo ">>> 检测到环境不完整，准备下载驱动包进行修复..."

# 安装 wget (如果镜像里没有)
if ! command -v wget &> /dev/null; then
    apt-get update && apt-get install -y wget
fi

INSTALLER="NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
DOWNLOAD_URL="${NVIDIA_URL_BASE}/${DRIVER_VERSION}/${INSTALLER}"

WORKDIR=$(mktemp -d)
cd ${WORKDIR}

echo ">>> 下载驱动: ${DOWNLOAD_URL}"
# 尝试下载，如果失败则输出错误
if ! wget -q --show-progress "${DOWNLOAD_URL}"; then
    echo "Error: 下载失败！请检查网络或确认该驱动版本是否有公开下载链接。"
    exit 1
fi

echo ">>> 解压驱动包 (Extract only)..."
chmod +x ${INSTALLER}
./${INSTALLER} --extract-only --target extract_dir > /dev/null

echo ">>> 补全缺失文件到 ${LIB_DIR}..."
cd extract_dir
echo  ${DRIVER_VERSION}
echo extract_dir
# 复制 gpucomp 和 glcore 以及 eglcore
cp libnvidia-gpucomp.so.${DRIVER_VERSION} ${LIB_DIR}/ || echo "Warning: 包内未找到 gpucomp (可能是旧版驱动)"
cp libnvidia-glcore.so.${DRIVER_VERSION} ${LIB_DIR}/
cp libnvidia-eglcore.so.${DRIVER_VERSION} ${LIB_DIR}/
# 也可以根据需要补全 TLS 库
cp libnvidia-tls.so.${DRIVER_VERSION} ${LIB_DIR}/ 2>/dev/null || true

# 清理临时文件
cd /
rm -rf ${WORKDIR}
echo ">>> 文件修补完成。"
# else
#     echo ">>> 关键库文件已存在，跳过下载。"
# fi

# 5. 重建动态链接库缓存 (至关重要)
echo ">>> 刷新 ldconfig..."
ldconfig

# 6. 设置环境变量
# 注意：脚本运行结束后，这些变量在父Shell会失效。
# 建议在脚本末尾执行用户的命令，或者 source 这个脚本。
export EGL_PLATFORM=surfaceless
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=all
export LD_LIBRARY_PATH=${LIB_DIR}:${LD_LIBRARY_PATH}

echo ">>> [Auto-Fix-EGL] 环境配置成功！"