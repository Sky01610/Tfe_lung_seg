import os
import sys

import SimpleITK as sitk


def convert_dicom_to_nifti(dicom_dir: str, output_nii_path: str) -> bool:
    """
    使用 SimpleITK 将 DICOM 序列转换为 NIfTI (.nii.gz) 格式。
    SimpleITK 会在底层自动处理并保留 Origin, Spacing, Direction 等空间几何信息。
    """
    print(f"正在扫描 DICOM 目录: {dicom_dir}")

    if not os.path.exists(dicom_dir):
        print(f"错误: 找不到目录 {dicom_dir}")
        return False

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir) or []

    if not series_ids:
        print("错误: 在指定目录中未找到任何 DICOM 序列！请检查目录路径。")
        return False

    if len(series_ids) > 1:
        print(f"提示: 发现 {len(series_ids)} 个不同的图像序列，将选择层数最多的序列。")
        series_id = max(
            series_ids,
            key=lambda sid: len(reader.GetGDCMSeriesFileNames(dicom_dir, sid)),
        )
    else:
        series_id = series_ids[0]

    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)
    reader.SetFileNames(dicom_names)

    try:
        print("正在读取并重建 3D 图像...")
        image = reader.Execute()

        print("\n--- 图像空间几何信息 ---")
        print(f"尺寸 (Size):          {image.GetSize()}")
        print(f"像素间距 (Spacing):   {image.GetSpacing()}")
        print(f"物理原点 (Origin):    {image.GetOrigin()}")
        print(f"方向矩阵 (Direction): {image.GetDirection()}")
        print("------------------------\n")

        output_dir = os.path.dirname(output_nii_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"正在保存为 NIfTI 文件: {output_nii_path}")
        sitk.WriteImage(image, output_nii_path)
        print("转换成功。")
        return True

    except Exception as e:
        print(f"转换过程中发生错误: {e}")
        return False

if __name__ == "__main__":
    # 使用示例
    # 请在这里替换为你实际的 DICOM 文件夹路径和期望保存的 nii.gz 文件路径
    input_dicom_directory = r"F:\CT_5_20\2025121244421967\2025121244421967" 
    output_nifti_file = r"d:\work\bronchus_client\2025121244421967.nii.gz"
    
    # 如果你想直接在终端运行测试，可以把上面两行改成真实的路径，然后取消下面的注释
    convert_dicom_to_nifti(input_dicom_directory, output_nifti_file)
    
    print("脚本已加载。请修改代码底部的路径并调用 convert_dicom_to_nifti 函数。")