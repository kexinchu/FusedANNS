/***************************
@Author: xxx
@Contact: xxx@xxxx.com
@File: meta_data.h
@Time: 2022/5/2 11:18
@Desc: 
***************************/

#ifndef GRAPHANNS_META_DATA_H
#define GRAPHANNS_META_DATA_H

#include <type_traits>
#include <vector>
#include "../data_objects_define.h"

template<typename T = float>
struct MetaData {
    T *data = nullptr;
    IDType num = 0;
    unsigned dim = 0;
    std::string file_path;

    CStatus norm(T *vec, const unsigned n, const unsigned d) {
        if constexpr (std::is_integral<T>::value) {
            return CStatus();
        }
        for (size_t i = 0; i < n; i++) {
            float vector_norm = 0;
            for (size_t j = 0; j < d; j++) {
                vector_norm += vec[i * d + j] * vec[i * d + j];
            }
            vector_norm = std::sqrt(vector_norm);
            if (vector_norm <= 0) {
                continue;
            }
            for (size_t j = 0; j < d; j++) {
                vec[i * d + j] /= vector_norm;
            }
        }
        return CStatus();
    }

    CStatus load(const std::string& path, const unsigned is_norm) {
        std::ifstream in(path.data(), std::ios::binary);
        if (!in.is_open()) {
            return CStatus(path + " open file error!");
        }
        unsigned dim_val_size = sizeof(unsigned);
        bool load_as_int = false;
        if constexpr (std::is_same<T, float>::value) {
            const std::string ext = ".ivecs";
            if (path.size() >= ext.size() &&
                path.compare(path.size() - ext.size(), ext.size(), ext) == 0) {
                load_as_int = true;
            }
        }
        if (load_as_int) {
            in.read((char *) &dim, dim_val_size);
            in.seekg(0, std::ios::end);
            std::ios::pos_type ss = in.tellg();
            auto f_size = (size_t) ss;
            num = (IDType) (f_size / (dim * sizeof(int) + dim_val_size));
            data = new T[(size_t)num * (size_t)dim];
            std::vector<int> tmp(dim);
            in.seekg(0, std::ios::beg);
            for (size_t i = 0; i < num; i++) {
                in.seekg(dim_val_size, std::ios::cur);
                in.read((char *) tmp.data(), dim * sizeof(int));
                for (size_t j = 0; j < dim; j++) {
                    data[i * dim + j] = static_cast<T>(tmp[j]);
                }
            }
        } else {
            in.read((char *) &dim, dim_val_size);
            in.seekg(0, std::ios::end);
            std::ios::pos_type ss = in.tellg();
            auto f_size = (size_t) ss;
            num = (IDType) (f_size / (dim * sizeof(T) + dim_val_size));
            data = new T[(size_t)num * (size_t)dim];

            in.seekg(0, std::ios::beg);
            for (size_t i = 0; i < num; i++) {
                in.seekg(dim_val_size, std::ios::cur);
                in.read((char *) (data + i * dim), dim * sizeof(T));
            }
        }
        in.close();
        file_path = path;
        if (is_norm) {
            norm(data, num, dim);
            printf("[EXEC] normalize vector complete!\n");
        }
        return CStatus();
    }

    virtual ~MetaData() {
        if (data) {
            delete[] data;
            data = nullptr;
        }
    }
};

#endif //GRAPHANNS_META_DATA_H
