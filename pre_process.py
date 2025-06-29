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
import shutil
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

class SafeBatchImageExtractor:
    """安全的批量图像提取器 - 解决文件消失问题"""
    
    def __init__(self, img_dir, pickle_file):
        self.img_dir = img_dir
        self.pickle_file = pickle_file
        self.progress_file = pickle_file.replace('.pkl', '_progress.json')
        self.backup_file = pickle_file.replace('.pkl', '_backup.pkl')
        
        # 确保pickle文件的目录存在
        pickle_dir = os.path.dirname(self.pickle_file)
        if pickle_dir:
            ensure_folder(pickle_dir)
        
    def extract_all_images(self, idx_path, rec_path, batch_size=1000, save_interval=1000):
        """批量提取所有图像 - 更频繁的安全保存"""
        print(f"目标图像目录: {self.img_dir}")
        print(f"目标pickle文件: {self.pickle_file}")
        print(f"备份文件: {self.backup_file}")
        
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
                        
                        # 更频繁的安全保存 - 每1000个样本保存一次
                        current_time = time.time()
                        if (i + 1) % save_interval == 0 or current_time - last_save_time > 60:  # 每分钟保存一次
                            print(f"\n正在保存进度... (已处理 {success_count} 个样本)")
                            
                            # 安全保存
                            self._safe_save_all(i + 1, samples, class_ids)
                            last_save_time = current_time
                            
                            # 更新进度条信息
                            pbar.set_postfix({
                                'Success': success_count,
                                'Errors': error_count,
                                'Classes': len(class_ids),
                                'Last_Save': 'OK'
                            })
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        error_count += 1
                        if error_count <= 20:  # 只打印前20个错误
                            tqdm.write(f"处理第{i}个样本时出错: {e}")
                        pbar.update(1)
                        continue
                        
        except KeyboardInterrupt:
            print("\n🛑 用户中断，正在安全保存当前进度...")
            self._safe_save_all(i, samples, class_ids)
            print("✅ 进度已安全保存，下次运行将从此处继续")
            return samples
            
        except Exception as e:
            print(f"\n❌ 提取过程中出错: {e}")
            self._safe_save_all(i, samples, class_ids)
        
        # 最终保存
        print(f"\n🎉 提取完成！")
        print(f"成功处理: {success_count}")
        print(f"失败: {error_count}")
        if success_count + error_count > 0:
            print(f"成功率: {success_count/(success_count+error_count)*100:.2f}%")
        
        # 最终安全保存
        self._safe_save_all(total_records, samples, class_ids)
        self._print_statistics(samples, class_ids)
        
        # 清理进度文件
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
        
        return samples
    
    def _safe_save_all(self, current_idx, samples, class_ids):
        """安全保存所有数据 - 多重备份策略"""
        try:
            # 1. 保存进度文件
            self._save_progress(current_idx, samples, class_ids)
            
            # 2. 保存主pickle文件
            self._safe_save_pickle(samples, self.pickle_file)
            
            # 3. 保存备份文件
            self._safe_save_pickle(samples, self.backup_file)
            
            print(f"✅ 数据已安全保存 (主文件 + 备份)")
            
        except Exception as e:
            print(f"❌ 安全保存失败: {e}")
    
    def _safe_save_pickle(self, samples, filepath):
        """安全保存pickle文件 - 直接写入，立即刷新"""
        try:
            print(f"保存 {len(samples)} 个样本到 {filepath}")
            
            # 直接写入，不使用临时文件
            with open(filepath, 'wb') as f:
                pickle.dump(samples, f)
                f.flush()  # 强制刷新到磁盘
                os.fsync(f.fileno())  # 确保写入磁盘
            
            # 验证文件存在且有内容
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"✅ 文件保存成功: {filepath} ({size} 字节)")
            else:
                print(f"❌ 警告: 文件保存后不存在: {filepath}")
                
        except Exception as e:
            print(f"❌ 保存pickle失败: {e}")
            # 如果保存失败，至少保存一个JSON版本作为备用
            try:
                json_file = filepath.replace('.pkl', '.json')
                with open(json_file, 'w') as f:
                    json.dump(samples[:1000], f)  # 只保存前1000个样本的JSON
                print(f"✅ 备用JSON文件已保存: {json_file}")
            except:
                pass
    
    def _load_progress(self):
        """加载之前的进度"""
        # 优先从主文件加载
        for pickle_path in [self.pickle_file, self.backup_file]:
            if os.path.exists(pickle_path):
                try:
                    with open(pickle_path, 'rb') as f:
                        samples = pickle.load(f)
                    print(f"✅ 从 {pickle_path} 加载了 {len(samples)} 个样本")
                    
                    # 从进度文件获取开始索引
                    start_idx = 0
                    if os.path.exists(self.progress_file):
                        try:
                            with open(self.progress_file, 'r') as f:
                                progress = json.load(f)
                            start_idx = progress.get('last_idx', len(samples))
                        except:
                            start_idx = len(samples)
                    else:
                        start_idx = len(samples)
                    
                    # 重建class_ids
                    class_ids = set([s['label'] for s in samples])
                    
                    print(f"继续从索引 {start_idx} 开始")
                    return start_idx, samples, class_ids
                    
                except Exception as e:
                    print(f"加载 {pickle_path} 失败: {e}")
                    continue
        
        print("没有找到之前的进度，从头开始")
        return 0, [], set()
    
    def _save_progress(self, current_idx, samples, class_ids):
        """保存当前进度"""
        progress = {
            'last_idx': current_idx,
            'total_samples': len(samples),
            'total_classes': len(class_ids),
            'timestamp': time.time(),
            'pickle_file': self.pickle_file,
            'backup_file': self.backup_file
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
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
    
    def _print_statistics(self, samples, class_ids):
        """打印统计信息"""
        print(f'\n=== 📊 最终统计 ===')
        print(f'总样本数: {len(samples)}')
        print(f'类别数: {len(class_ids)}')
        
        if class_ids:
            class_ids_list = list(class_ids)
            print(f'最大类别ID: {max(class_ids_list)}')
            print(f'最小类别ID: {min(class_ids_list)}')
        
        # 验证文件存在
        print(f'\n=== 📁 文件验证 ===')
        for file_path in [self.pickle_file, self.backup_file]:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f'✅ {file_path}: {size} 字节')
            else:
                print(f'❌ {file_path}: 不存在')

def main():
    """主函数 - 安全批量提取"""
    
    print("=== 🚀 安全批量图像提取器 ===")
    print(f"目标目录: {IMG_DIR}")
    print(f"元数据文件: {pickle_file}")
    
    # 检查配置
    print(f"\n=== 🔍 配置检查 ===")
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
    
    # 创建安全提取器
    extractor = SafeBatchImageExtractor(IMG_DIR, pickle_file)
    
    # 开始批量提取 - 更频繁的保存
    print("\n🏃‍♂️ 开始安全批量提取...")
    print("💡 提示: 文件每1000个样本自动保存一次，Ctrl+C 可安全中断")
    
    samples = extractor.extract_all_images(
        path_imgidx, 
        path_imgrec,
        batch_size=1000,      # 批次大小
        save_interval=1000    # 每1000个样本保存一次 (更频繁)
    )
    
    if len(samples) > 0:
        print(f"\n🎉 批量提取完成！总共提取了 {len(samples)} 个样本")
        
        # 最终验证
        for file_path in [pickle_file, pickle_file.replace('.pkl', '_backup.pkl')]:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        test_samples = pickle.load(f)
                    print(f"✅ {file_path} 验证成功，包含 {len(test_samples)} 个样本")
                except Exception as e:
                    print(f"❌ {file_path} 验证失败: {e}")
            else:
                print(f"❌ 警告: {file_path} 不存在")
                
    else:
        print("❌ 没有成功提取任何样本")

if __name__ == "__main__":
    main()