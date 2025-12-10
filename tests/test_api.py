"""
AI Detection Service 测试脚本
测试各个 API 端点的基本功能
"""
import requests
import time
import sys
from pathlib import Path

# 服务地址配置
BASE_URL = "http://localhost:5000"


def test_root():
    """测试根路由"""
    print("\n" + "=" * 50)
    print("测试: 根路由 /")
    print("=" * 50)

    try:
            response = requests.get(f"{BASE_URL}/")
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                print(f"响应: {response.json()}")
                print("✅ 根路由测试通过")
                return True
            else:
                print("❌ 根路由测试失败")
                return False
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            return False


    def test_health():
        """测试健康检查端点"""
        print("\n" + "=" * 50)
        print("测试: 健康检查 /api/health")
        print("=" * 50)

        try:
            response = requests.get(f"{BASE_URL}/api/health")
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                print(f"响应: {response.json()}")
                print("✅ 健康检查测试通过")
                return True
            else:
                print("❌ 健康检查测试失败")
                return False
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            return False


    def test_model_info():
    """测试模型信息端点"""
    print("\n" + "=" * 50)
    print("测试: 模型信息 /api/model/info")
    print("=" * 50)

    try:
        response = requests.get(f"{BASE_URL}/api/model/info")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print(f"响应: {response.json()}")
            print("✅ 模型信息测试通过")
            return True
        else:
            print("❌ 模型信息测试失败")
            return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False


def test_detect_image(image_path: str):
    """测试单张图片检测"""
    print("\n" + "=" * 50)
    print(f"测试: 图片检测 /api/detect")
    print(f"图片: {image_path}")
    print("=" * 50)

    if not Path(image_path).exists():
        print(f"❌ 图片文件不存在: {image_path}")
        return False

    try:
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/api/detect", files=files)
            elapsed = time.time() - start_time

        print(f"状态码: {response.status_code}")
        print(f"耗时: {elapsed:.3f}秒")

        if response.status_code == 200:
            result = response.json()
            print(f"检测结果:")
            print(f"  - 成功: {result.get('success', False)}")
            print(f"  - 检测数量: {len(result.get('detections', []))}")

            # 打印检测到的缺陷
            for i, det in enumerate(result.get('detections', [])[:5]):
                print(f"  - 缺陷{i+1}: {det.get('class_name', 'unknown')} "
                      f"(置信度: {det.get('confidence', 0):.2%})")

            if len(result.get('detections', [])) > 5:
                print(f"  - ... 还有 {len(result.get('detections', [])) - 5} 个检测结果")

            print("✅ 图片检测测试通过")
            return True
        else:
            print(f"响应: {response.text}")
            print("❌ 图片检测测试失败")
            return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False


def test_detect_batch(image_paths: list):
    """测试批量图片检测"""
    print("\n" + "=" * 50)
    print(f"测试: 批量检测 /api/detect/batch")
    print(f"图片数量: {len(image_paths)}")
    print("=" * 50)

    valid_paths = [p for p in image_paths if Path(p).exists()]
    if not valid_paths:
        print("❌ 没有有效的图片文件")
        return False

    try:
        files = []
        for path in valid_paths:
            files.append(('files', (Path(path).name, open(path, 'rb'), 'image/jpeg')))

        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/detect/batch", files=files)
        elapsed = time.time() - start_time

        for _, (_, f, _) in files:
            f.close()

        print(f"状态码: {response.status_code}")
        print(f"耗时: {elapsed:.3f}秒")

        if response.status_code == 200:
            result = response.json()
            print(f"批量检测结果: {result.get('success', False)}")
            print("✅ 批量检测测试通过")
            return True
        else:
            print(f"响应: {response.text}")
            print("❌ 批量检测测试失败")
            return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False


def test_docs():
    """测试API文档端点"""
    print("\n" + "=" * 50)
    print("测试: API文档 /docs")
    print("=" * 50)

    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("✅ API文档可访问")
            return True
        else:
            print("❌ API文档访问失败")
            return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False


def run_all_tests(test_image: str = None):
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("    AI Detection Service 测试开始")
    print("=" * 60)
    print(f"服务地址: {BASE_URL}")

    results = {
        "根路由": test_root(),
        "健康检查": test_health(),
        "模型信息": test_model_info(),
        "API文档": test_docs(),
    }

    if test_image and Path(test_image).exists():
        results["图片检测"] = test_detect_image(test_image)
    else:
        print("\n⚠️  未提供测试图片，跳过图片检测测试")
        print("   使用方法: python test_api.py <图片路径>")

    print("\n" + "=" * 60)
    print("    测试总结")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")

    print("-" * 60)
    print(f"  总计: {passed}/{total} 测试通过")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    test_image = sys.argv[1] if len(sys.argv) > 1 else None
    success = run_all_tests(test_image)
    sys.exit(0 if success else 1)
