/*
 * test_soa_performance.cpp
 * 
 * 测试SSF_GPU_SoA与原始SSF_GPU的性能对比
 * 编译: nvcc -I./include -I./src test_soa_performance.cpp src/SSF_Driver.cc src/SSF_FV_SoA_t.cc -o test_soa
 */

#include <iostream>
#include <chrono>
#include <cstring>
#include "SSF_Driver.h"
#include "SSF_FV_SoA_t.h"

// 简单的计时器类
class Timer {
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // 返回毫秒
    }
private:
    std::chrono::high_resolution_clock::time_point start_time;
};

// 创建测试数据
void generate_test_data(real8* nv, SSF_PV_t* pv, int nn, int np) {
    // 生成随机节点位置
    for (int i = 0; i < nn * 3; i++) {
        nv[i] = static_cast<real8>(rand()) / RAND_MAX * 100.0;
    }
    
    // 生成段对信息
    for (int i = 0; i < np; i++) {
        pv[i].n1 = rand() % nn;
        pv[i].n2 = rand() % nn;
        pv[i].n3 = rand() % nn;
        pv[i].n4 = rand() % nn;
        
        // 生成burgers向量
        for (int j = 0; j < 3; j++) {
            pv[i].b1[j] = static_cast<real8>(rand()) / RAND_MAX * 2.0 - 1.0;
            pv[i].b3[j] = static_cast<real8>(rand()) / RAND_MAX * 2.0 - 1.0;
        }
    }
}

// 验证结果一致性
bool verify_results(const SSF_FV_t* fv1, const SSF_FV_t* fv2, int np, double tolerance = 1e-10) {
    for (int i = 0; i < np; i++) {
        for (int j = 0; j < 3; j++) {
            if (std::abs(fv1[i].f1[j] - fv2[i].f1[j]) > tolerance ||
                std::abs(fv1[i].f2[j] - fv2[i].f2[j]) > tolerance ||
                std::abs(fv1[i].f3[j] - fv2[i].f3[j]) > tolerance ||
                std::abs(fv1[i].f4[j] - fv2[i].f4[j]) > tolerance) {
                std::cout << "结果不一致: pair=" << i << ", comp=" << j << std::endl;
                std::cout << "  原始: f1=" << fv1[i].f1[j] << ", SoA: f1=" << fv2[i].f1[j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    std::cout << "=== SSF SoA 性能测试 ===" << std::endl;
    
    // 测试参数
    const int nn = 1000;    // 节点数
    const int np = 5000;    // 段对数
    const int iterations = 10;
    
    // 分配内存
    real8* nv = new real8[nn * 3];
    SSF_PV_t* pv = new SSF_PV_t[np];
    SSF_FV_t* fv_original = new SSF_FV_t[np];
    SSF_FV_t* fv_soa = new SSF_FV_t[np];
    
    // 生成测试数据
    srand(42); // 固定种子确保可重复性
    generate_test_data(nv, pv, nn, np);
    
    std::cout << "测试配置:" << std::endl;
    std::cout << "  节点数: " << nn << std::endl;
    std::cout << "  段对数: " << np << std::endl;
    std::cout << "  迭代次数: " << iterations << std::endl;
    std::cout << std::endl;
    
    // 初始化SSF系统
    try {
        // 模拟Home_t结构的基本参数
        // 注意: 实际使用中需要正确初始化这些参数
        SSF_Initialize_Iso(
            2.5,    // core radius
            1.0,    // shear modulus
            0.3,    // poisson ratio
            0.1,    // ecrit
            0, 0, 0,  // PBC flags
            100.0, 100.0, 100.0,  // box size
            128,    // threads per block
            64*1024,  // pairs per block
            4       // streams
        );
        
        Timer timer;
        double time_original = 0.0;
        double time_soa = 0.0;
        
        std::cout << "开始性能测试..." << std::endl;
        
        // 测试原始版本
        std::cout << "测试原始AoS版本..." << std::endl;
        for (int i = 0; i < iterations; i++) {
            memset(fv_original, 0, np * sizeof(SSF_FV_t));
            timer.start();
            SSF_GPU(fv_original, nv, pv, nn, np, 0);  // mode=0 (isotropic)
            time_original += timer.stop();
        }
        time_original /= iterations;
        
        // 测试SoA版本
        std::cout << "测试新SoA版本..." << std::endl;
        for (int i = 0; i < iterations; i++) {
            memset(fv_soa, 0, np * sizeof(SSF_FV_t));
            timer.start();
            SSF_GPU_SoA(fv_soa, nv, pv, nn, np, 0);  // mode=0 (isotropic)
            time_soa += timer.stop();
        }
        time_soa /= iterations;
        
        // 验证结果一致性
        std::cout << "\n验证结果一致性..." << std::endl;
        if (verify_results(fv_original, fv_soa, np)) {
            std::cout << "✓ 结果验证通过!" << std::endl;
        } else {
            std::cout << "✗ 结果验证失败!" << std::endl;
        }
        
        // 输出性能结果
        std::cout << "\n=== 性能测试结果 ===" << std::endl;
        std::cout << "原始AoS版本平均时间: " << time_original << " ms" << std::endl;
        std::cout << "新SoA版本平均时间:   " << time_soa << " ms" << std::endl;
        
        if (time_original > 0) {
            double speedup = time_original / time_soa;
            std::cout << "性能提升倍数: " << speedup << "x" << std::endl;
            
            if (speedup > 1.0) {
                std::cout << "🎉 SoA版本性能提升 " << ((speedup - 1.0) * 100) << "%!" << std::endl;
            } else {
                std::cout << "⚠️  SoA版本性能下降 " << ((1.0 - speedup) * 100) << "%!" << std::endl;
            }
        }
        
        // 内存使用情况分析
        std::cout << "\n=== 内存使用分析 ===" << std::endl;
        size_t aos_memory = np * sizeof(SSF_FV_t);
        size_t soa_memory = 12 * np * sizeof(real8);  // 12个分量数组
        
        std::cout << "AoS内存使用: " << aos_memory << " bytes" << std::endl;
        std::cout << "SoA内存使用: " << soa_memory << " bytes" << std::endl;
        std::cout << "内存开销比例: " << (double)soa_memory / aos_memory << "x" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "测试失败: " << e.what() << std::endl;
    }
    
    // 清理内存
    delete[] nv;
    delete[] pv;
    delete[] fv_original;
    delete[] fv_soa;
    
    SSF_Free();
    
    std::cout << "\n测试完成!" << std::endl;
    return 0;
}

