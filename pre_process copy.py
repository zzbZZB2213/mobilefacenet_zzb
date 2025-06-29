import os
import pickle
import struct
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import json
import time
from config import path_imgidx, path_imgrec, IMG_DIR, pickle_file

def ensure_folder(folder):
    """确保文件夹存在"""
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"创建目录: {folder}")

class RecordIOReader:
    """针对文本索引格式的RecordIO读取器"""
    
    def __init__(self, idx_path, rec_path):
        self.idx_path = idx_path
        self.rec_path = rec_path
        
        # 检查文件是否存在
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"索引文件不存在: {idx_path}")
        if not os.path.exists(rec_path):
            raise FileNotFoundError(f"记录文件不存在: {rec_path}")
            
        self.offsets = self._load_text_indices()
        
    def _load_text_indices(self):
        """加载文本格式的索引文件"""
        offsets = []
        
        print("正在加载索引文件...")
        with open(self.idx_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        # 索引文件格式: 记录ID\t偏移量
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            offset = int(parts[1])
                            offsets.append(offset)
                    except ValueError as e:
                        print(f"解析第{line_num+1}行时出错: {line}, 错误: {e}")
                        continue
        
        print(f"成功加载 {len(offsets)} 个偏移量")
        return offsets
        
    def __len__(self):
        return len(self.offsets)
    
    def read_idx(self, idx):
        """读取指定索引的记录"""
        if idx <= 0 or idx > len(self.offsets):
            raise IndexError(f"Index {idx} out of range [1, {len(self.offsets)}]")
        
        # 获取当前记录的偏移量
        current_offset = self.offsets[idx - 1]
        
        # 计算记录长度
        if idx < len(self.offsets):
            next_offset = self.offsets[idx]
            length = next_offset - current_offset
        else:
            # 最后一个记录，长度到文件末尾
            file_size = os.path.getsize(self.rec_path)
            length = file_size - current_offset
        
        # 读取数据
        with open(self.rec_path, 'rb') as f:
            f.seek(current_offset)
            data = f.read(length)
        
        if len(data) == 0:
            raise ValueError(f"读取数据为空，偏移量: {current_offset}, 长度: {length}")
            
        return self._unpack_record(data, idx)
    
    def _unpack_record(self, data, idx):
        """解包记录数据"""
        if len(data) < 16:
            raise ValueError(f"记录数据太短: {len(data)} 字节")
        
        # 查找JPEG文件头 (FF D8 FF)
        jpeg_start = -1
        for i in range(len(data) - 3):
            if data[i] == 0xFF and data[i+1] == 0xD8 and data[i+2] == 0xFF:
                jpeg_start = i
                break
        
        if jpeg_start == -1:
            # 查找PNG头: 89 50 4E 47
            for i in range(len(data) - 4):
                if (data[i] == 0x89 and data[i+1] == 0x50 and 
                    data[i+2] == 0x4E and data[i+3] == 0x47):
                    jpeg_start = i
                    break
        
        if jpeg_start == -1:
            raise ValueError(f"找不到有效的图像数据头，记录 {idx}")
        
        # 提取header和图像数据
        header_data = data[:jpeg_start] if jpeg_start > 0 else b""
        image_data = data[jpeg_start:]
        
        # 解析header
        header = self._parse_header(header_data, idx)
        
        return header, image_data
    
    def _parse_header(self, header_data, idx):
        """解析header数据"""
        class Header:
            def __init__(self, label=0):
                self.label = label
        
        if len(header_data) == 0:
            return Header(idx % 10000)  # 使用索引作为临时label
            
        # 尝试从header中提取label
        try:
            # 查找header中的数值
            for i in range(0, len(header_data) - 4, 1):
                try:
                    value = struct.unpack('<I', header_data[i:i+4])[0]
                    if 0 <= value <= 1000000:  # 合理的label范围
                        return Header(value)
                except:
                    continue
        except:
            pass
        
        # 使用idx作为fallback label
        return Header(idx % 10000)

class BatchImageExtractor:
    """批量图像提取器"""
    
    def __init__(self, img_dir, pickle_file):
        self.img_dir = img_dir
        self.pickle_file = pickle_file
        self.progress_file = pickle_file.replace('.pkl', '_progress.json')
        
        # 确保pickle文件的目录存在
        pickle_dir = os.path.dirname(self.pickle_file)
        if pickle_dir:
            ensure_folder(pickle_dir)
        
    def extract_all_images(self, idx_path, rec_path, batch_size=1000, save_interval=10000):
        """批量提取所有图像"""
        print(f"目标图像目录: {self.img_dir}")
        print(f"目标pickle文件: {self.pickle_file}")
        
        ensure_folder(self.img_dir)
        
        # 初始化读取器
        print("初始化RecordIO读取器...")
        try:
            reader = RecordIOReader(idx_path, rec_path)
        except Exception as e:
            print(f"无法初始化RecordIO读取器: {e}")
            return []
        
        total_records = len(reader)
        print(f"总记录数: {total_records}")
        
        # 检查是否有之前的进度
        start_idx, samples, class_ids = self._load_progress()
        
        print(f"从索引 {start_idx} 开始提取...")
        
        success_count = len(samples)
        error_count = 0
        last_save_time = time.time()
        
        try:
            # 使用tqdm显示进度
            with tqdm(total=total_records, initial=start_idx, desc="提取图像") as pbar:
                for i in range(start_idx, total_records):
                    try:
                        # 读取记录
                        header, image_data = reader.read_idx(i + 1)
                        
                        # 解码图像
                        img = self._decode_image(image_data)
                        if img is None:
                            error_count += 1
                            pbar.update(1)
                            continue
                        
                        # 转换为OpenCV格式 (BGR)
                        img_cv = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
                        
                        # 获取标签
                        label = int(header.label)
                        class_ids.add(label)
                        
                        # 保存图像
                        filename = f'{i}.jpg'
                        filepath = os.path.join(self.img_dir, filename)
                        success_write = cv.imwrite(filepath, img_cv)
                        
                        if not success_write:
                            print(f"警告: 无法保存图像 {filepath}")
                            error_count += 1
                            pbar.update(1)
                            continue
                        
                        # 添加到样本列表
                        samples.append({'img': filename, 'label': label})
                        success_count += 1
                        
                        # 定期保存进度
                        current_time = time.time()
                        if (i + 1) % save_interval == 0 or current_time - last_save_time > 300:  # 每5分钟保存一次
                            self._save_progress(i + 1, samples, class_ids)
                            self._save_metadata(samples)
                            last_save_time = current_time
                            print(f"已保存进度到: {self.pickle_file}")
                            
                            # 更新进度条信息
                            pbar.set_postfix({
                                'Success': success_count,
                                'Errors': error_count,
                                'Classes': len(class_ids)
                            })
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        error_count += 1
                        if error_count <= 20:  # 只打印前20个错误
                            tqdm.write(f"处理第{i}个样本时出错: {e}")
                        pbar.update(1)
                        continue
                        
        except KeyboardInterrupt:
            print("\n用户中断，保存当前进度...")
            self._save_progress(i, samples, class_ids)
            self._save_metadata(samples)
        except Exception as e:
            print(f"\n提取过程中出错: {e}")
            self._save_progress(i, samples, class_ids)
            self._save_metadata(samples)
        
        # 最终保存
        print(f"\n提取完成！")
        print(f"成功处理: {success_count}")
        print(f"失败: {error_count}")
        if success_count + error_count > 0:
            print(f"成功率: {success_count/(success_count+error_count)*100:.2f}%")
        
        # 强制保存最终结果
        try:
            self._save_metadata(samples)
            print(f"✅ 最终数据已保存到: {self.pickle_file}")
            
            # 验证文件是否真的被创建
            if os.path.exists(self.pickle_file):
                file_size = os.path.getsize(self.pickle_file)
                print(f"✅ Pickle文件大小: {file_size} 字节")
            else:
                print(f"❌ 警告: Pickle文件未创建!")
                
        except Exception as e:
            print(f"❌ 保存最终数据时出错: {e}")
        
        self._print_statistics(samples, class_ids)
        
        # 清理进度文件
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
        
        return samples
    
    def _load_progress(self):
        """加载之前的进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                
                start_idx = progress.get('last_idx', 0)
                
                # 加载已有的samples
                if os.path.exists(self.pickle_file):
                    with open(self.pickle_file, 'rb') as f:
                        samples = pickle.load(f)
                    print(f"✅ 成功加载已有的pickle文件: {len(samples)} 个样本")
                else:
                    samples = []
                
                # 重建class_ids
                class_ids = set([s['label'] for s in samples])
                
                print(f"发现之前的进度: 已处理 {len(samples)} 个样本，从索引 {start_idx} 继续")
                return start_idx, samples, class_ids
                
            except Exception as e:
                print(f"加载进度失败: {e}")
        
        return 0, [], set()
    
    def _save_progress(self, current_idx, samples, class_ids):
        """保存当前进度"""
        progress = {
            'last_idx': current_idx,
            'total_samples': len(samples),
            'total_classes': len(class_ids),
            'timestamp': time.time()
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            print(f"保存进度文件失败: {e}")
    
    def _decode_image(self, image_data):
        """解码图像数据"""
        if len(image_data) < 10:
            return None
            
        try:
            # 使用PIL解码
            img = Image.open(BytesIO(image_data))
            img = img.convert('RGB')
            return img
        except:
            try:
                # 使用OpenCV解码
                img_array = np.frombuffer(image_data, dtype=np.uint8)
                img_cv = cv.imdecode(img_array, cv.IMREAD_COLOR)
                if img_cv is not None and img_cv.size > 0:
                    img_rgb = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
                    return Image.fromarray(img_rgb)
            except:
                pass
        return None
    
    def _save_metadata(self, samples):
        """保存元数据"""
        try:
            print(f"正在保存 {len(samples)} 个样本到 {self.pickle_file}")
            
            # 确保目录存在
            pickle_dir = os.path.dirname(self.pickle_file)
            if pickle_dir and not os.path.exists(pickle_dir):
                os.makedirs(pickle_dir)
                print(f"创建目录: {pickle_dir}")
            
            # 保存到临时文件，然后重命名（原子操作）
            temp_file = self.pickle_file + '.tmp'
            with open(temp_file, 'wb') as file:
                pickle.dump(samples, file)
            
            # 重命名到最终文件
            os.rename(temp_file, self.pickle_file)
            print(f"✅ 成功保存到: {self.pickle_file}")
            
        except Exception as e:
            print(f"❌ 保存元数据失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_statistics(self, samples, class_ids):
        """打印统计信息"""
        print(f'\n=== 提取统计 ===')
        print(f'总样本数: {len(samples)}')
        print(f'类别数: {len(class_ids)}')
        
        if class_ids:
            class_ids_list = list(class_ids)
            print(f'最大类别ID: {max(class_ids_list)}')
            print(f'最小类别ID: {min(class_ids_list)}')
        
        # 计算每个类别的样本数
        label_counts = {}
        for sample in samples:
            label = sample['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        if label_counts:
            print(f'平均每类样本数: {len(samples)/len(label_counts):.1f}')
            print(f'最多样本的类别: {max(label_counts.values())} 个样本')
            print(f'最少样本的类别: {min(label_counts.values())} 个样本')

def main():
    """主函数 - 批量提取"""
    
    print("=== 批量图像提取器 ===")
    print(f"目标目录: {IMG_DIR}")
    print(f"元数据文件: {pickle_file}")
    
    # 检查配置
    print(f"\n=== 配置检查 ===")
    print(f"索引文件: {path_imgidx}")
    print(f"记录文件: {path_imgrec}")
    print(f"索引文件存在: {os.path.exists(path_imgidx)}")
    print(f"记录文件存在: {os.path.exists(path_imgrec)}")
    
    if not os.path.exists(path_imgidx):
        print(f"❌ 错误: 索引文件不存在: {path_imgidx}")
        return
    
    if not os.path.exists(path_imgrec):
        print(f"❌ 错误: 记录文件不存在: {path_imgrec}")
        return
    
    # 检查写入权限
    try:
        pickle_dir = os.path.dirname(pickle_file)
        if pickle_dir:
            ensure_folder(pickle_dir)
        
        # 测试写入权限
        test_file = pickle_file + '.test'
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"✅ 目标目录有写入权限")
        
    except Exception as e:
        print(f"❌ 写入权限检查失败: {e}")
        return
    
    # 创建提取器
    extractor = BatchImageExtractor(IMG_DIR, pickle_file)
    
    # 开始批量提取
    print("\n开始批量提取...")
    samples = extractor.extract_all_images(
        path_imgidx, 
        path_imgrec,
        batch_size=1000,      # 批次大小
        save_interval=5000    # 每5000个样本保存一次进度
    )
    
    if len(samples) > 0:
        print(f"\n✅ 批量提取完成！总共提取了 {len(samples)} 个样本")
        
        # 最终验证pickle文件
        if os.path.exists(pickle_file):
            try:
                with open(pickle_file, 'rb') as f:
                    test_samples = pickle.load(f)
                print(f"✅ Pickle文件验证成功，包含 {len(test_samples)} 个样本")
            except Exception as e:
                print(f"❌ Pickle文件验证失败: {e}")
        else:
            print(f"❌ 警告: Pickle文件不存在: {pickle_file}")
            
    else:
        print("❌ 没有成功提取任何样本")

if __name__ == "__main__":
    main()
# import os
# import pickle

# import cv2 as cv
# import mxnet as mx
# from mxnet import recordio
# from tqdm import tqdm

# from config import path_imgidx, path_imgrec, IMG_DIR, pickle_file
# from utils import ensure_folder

# if __name__ == "__main__":
#     ensure_folder(IMG_DIR)
#     imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
#     # print(len(imgrec))

#     samples = []
#     class_ids = set()

#     # # %% 1 ~ 5179510

#     try:
#         for i in tqdm(range(10000000)):
#             # print(i)
#             header, s = recordio.unpack(imgrec.read_idx(i + 1))
#             img = mx.image.imdecode(s).asnumpy()
#             # print(img.shape)
#             img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
#             # print(header.label)
#             # print(type(header.label))
#             label = int(header.label)
#             class_ids.add(label)
#             filename = '{}.jpg'.format(i)
#             samples.append({'img': filename, 'label': label})
#             filename = os.path.join(IMG_DIR, filename)
#             cv.imwrite(filename, img)
#             # except KeyboardInterrupt:
#             #     raise
#     except Exception as err:
#         print(err)

#     with open(pickle_file, 'wb') as file:
#         pickle.dump(samples, file)

#     print('num_samples: ' + str(len(samples)))

#     class_ids = list(class_ids)
#     print(len(class_ids))
#     print(max(class_ids))
