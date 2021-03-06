//
// Created by Blank on 2021/1/19.
//

#pragma once

#include <NvInfer.h>

#include <cassert>
#include <vector>

#include "../cuda/bbox_decode.h"

using namespace nvinfer1;

#define RETINAPLATE_PLUGIN_NAME "RetinaPlateDecode"
#define RETINAPLATE_PLUGIN_VERSION "1"
#define RETINAPLATE_PLUGIN_NAMESPACE ""

namespace d2dev {
    namespace retinaplate {

        class DecodePlugin : public IPluginV2 {
            float _score_thresh;
            int _top_n;
            size_t _num_classes;
            size_t _num_anchors;
            std::vector<float> _anchors;
            std::vector<float> _box_weights;
            std::vector<float> _oks_weights;

        protected:
            void deserialize(void const *data, size_t length) {
                const char *d = static_cast<const char *>(data);
                read(d, _score_thresh);
                read(d, _top_n);
                read(d, _num_classes);
                read(d, _num_anchors);
                size_t box_weights_size;
                read(d, box_weights_size);
                while (box_weights_size--)
                {
                    float val;
                    read(d, val);
                    _box_weights.push_back(val);
                }
                size_t oks_weights_size;
                read(d, oks_weights_size);
                while (oks_weights_size--)
                {
                    float val;
                    read(d, val);
                    _oks_weights.push_back(val);
                }
                size_t anchors_size;
                read(d, anchors_size);
                while (anchors_size--)
                {
                    float val;
                    read(d, val);
                    _anchors.push_back(val);
                }
            }

            size_t getSerializationSize() const override {
                return sizeof(_score_thresh) + sizeof(_top_n) + sizeof(_num_classes) + sizeof(_num_anchors)
                       + sizeof(size_t) + sizeof(float) * _box_weights.size()
                       + sizeof(size_t) + sizeof(float) * _oks_weights.size()
                       + sizeof(size_t) + sizeof(float) * _anchors.size()
            }

            void serialize(void *buffer) const override {
                char *d = static_cast<char *>(buffer);
                write(d, _score_thresh);
                write(d, _top_n);
                write(d, _num_classes);
                write(d, _num_anchors);
                write(d, _box_weights.size());
                for (auto &val : _box_weights)
                {
                    write(d, val);
                }
                write(d, _oks_weights.size());
                for (auto &val : _oks_weights)
                {
                    write(d, val);
                }
                write(d, _anchors.size());
                for (auto &val : _anchors)
                {
                    write(d, val);
                }
            }

        public:
            DecodePlugin(
                    float score_thresh, int top_n,
                    std::vector<float> const &anchors,
                    std::vector<float> const &box_weights,
                    std::vector<float> const &oks_weights
            ) : _score_thresh(score_thresh), _top_n(top_n), _anchors(anchors), _box_weights(box_weights),
                _oks_weights(oks_weights) {}

            DecodePlugin(void const *data, size_t length) {
                this->deserialize(data, length);
            }

            const char *getPluginType() const override {
                return RETINAPLATE_PLUGIN_NAME;
            }

            const char *getPluginVersion() const override {
                return RETINAPLATE_PLUGIN_VERSION;
            }

            int getNbOutputs() const override {
                return 4;
            }

            Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override {
                assert(nbInputDims == 3);
                assert(index < this->getNbOutputs());
                return Dims3(_top_n * (index == 1 ? 4 : 1), 1, 1);
            }

            bool supportsFormat(DataType type, PluginFormat format) const override {
                return type == DataType::kFLOAT && format == PluginFormat::kLINEAR;
            }

            void configureWithFormat(
                    const Dims *inputDims, int nbInputs, const Dims *outputDims,
                    int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override {
                assert(type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kLINEAR);
                assert(nbInputs == 3);
                auto const &scores_dims = inputDims[0];
                auto const &boxes_dims = inputDims[1];
                auto const &keypoints_dims = inputDims[2];
                assert(scores_dims.d[0] == boxes_dims.d[0]);
                assert(scores_dims.d[0] == keypoints_dims.d[0]);
                assert(scores_dims.d[1] == boxes_dims.d[1]);
                assert(scores_dims.d[1] == keypoints_dims.d[1]);
                _num_anchors = scores_dims.d[1];
                _num_classes = scores_dims.d[2];
            }

            int initialize() override { return 0; }

            void terminate() override {}

            size_t getWorkspaceSize(int maxBatchSize) const override {
                static int size = -1;
                if (size < 0)
                {
                    size = decode(maxBatchSize, nullptr, nullptr, _num_anchors, _num_classes, _anchors,
                                  _box_weights, _oks_weights, _score_thresh, _top_n, nullptr, 0, nullptr);
                }
                return size;
            }

            int enqueue(int batchSize,
                        const void *const *inputs, void **outputs,
                        void *workspace, cudaStream_t stream) override {

                return decode(batchSize, inputs, outputs, _num_anchors, _num_classes, _anchors,
                              _box_weights, _oks_weights, _score_thresh, _top_n,
                              workspace, getWorkspaceSize(batchSize), stream);
            }

            void destroy() override {
                delete this;
            };

            const char *getPluginNamespace() const override {
                return RETINANET_PLUGIN_NAMESPACE;
            }

            void setPluginNamespace(const char *N) override {

            }

            IPluginV2 *clone() const override {
                return new DecodePlugin(_score_thresh, _top_n, _anchors, _box_weights, _oks_weights);
            }

        private:
            template<typename T>
            void write(char *&buffer, const T &val) const {
                *reinterpret_cast<T *>(buffer) = val;
                buffer += sizeof(T);
            }

            template<typename T>
            void read(const char *&buffer, T &val) {
                val = *reinterpret_cast<const T *>(buffer);
                buffer += sizeof(T);
            }
        };

        class DecodePluginCreator : public IPluginCreator {
        public:
            DecodePluginCreator() {}

            const char *getPluginName() const override {
                return RETINAPLATE_PLUGIN_NAME;
            }

            const char *getPluginVersion() const override {
                return RETINAPLATE_PLUGIN_VERSION;
            }

            const char *getPluginNamespace() const override {
                return RETINAPLATE_PLUGIN_NAMESPACE;
            }

            IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override {
                return new DecodePlugin(serialData, serialLength);
            }

            void setPluginNamespace(const char *N) override {}

            const PluginFieldCollection *getFieldNames() override { return nullptr; }

            IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override { return nullptr; }
        };

        REGISTER_TENSORRT_PLUGIN(DecodePluginCreator);
    }
}


#undef RETINAPLATE_PLUGIN_NAME
#undef RETINAPLATE_PLUGIN_VERSION
#undef RETINAPLATE_PLUGIN_NAMESPACE

