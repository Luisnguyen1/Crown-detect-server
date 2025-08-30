#!/usr/bin/env python3
"""
Test Script cho AI Crowd Detection Server
=========================================

Script này sẽ test server và AI detection với ảnh có sẵn
"""

import requests
import os
import json
import time

# Server config
SERVER_URL = "http://localhost:7860"

def test_upload_image(image_path):
    """Test upload ảnh và AI analysis"""
    if not os.path.exists(image_path):
        print(f"❌ File không tồn tại: {image_path}")
        return None
    
    print(f"📤 Uploading: {image_path}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{SERVER_URL}/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload thành công!")
            print(f"📁 Filename: {result['filename']}")
            
            # Check if original image was deleted
            if 'note' in result:
                print(f"🗑️ {result['note']}")
            
            # In kết quả AI nếu có
            if 'analysis' in result and result['analysis']:
                analysis = result['analysis']
                print(f"\n🤖 AI ANALYSIS RESULTS:")
                print(f"⏱️  Processing time: {analysis.get('processing_time', 'N/A')}")
                print(f"👥 Total people: {analysis.get('total_people', 0)}")
                
                if 'crowd_analysis' in analysis:
                    crowd = analysis['crowd_analysis']
                    print(f"🏃 Crowds detected: {crowd.get('total_crowds', 0)}")
                    print(f"👫 People in crowds: {crowd.get('total_people_in_crowds', 0)}")
                    print(f"🚶 Isolated people: {crowd.get('isolated_people', 0)}")
                    
                    # Chi tiết từng đám đông
                    for i, crowd_info in enumerate(crowd.get('crowds', [])):
                        print(f"  • Crowd {i+1}: {crowd_info['people_count']} people")
            
            return result
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_get_stats():
    """Test lấy thống kê"""
    try:
        response = requests.get(f"{SERVER_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"\n📊 SERVER STATS:")
            print(f"📸 Total images: {stats['stats']['total_images_received']}")
            print(f"🤖 Total analyses: {stats['stats']['total_analyses_completed']}")
            print(f"🔧 AI status: {stats['stats']['ai_detector_status']}")
            return stats
        else:
            print(f"❌ Stats failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return None

def test_get_latest():
    """Test lấy kết quả mới nhất"""
    try:
        response = requests.get(f"{SERVER_URL}/latest")
        if response.status_code == 200:
            result = response.json()
            print(f"\n📋 LATEST ANALYSIS:")
            print(f"📁 File: {result['filename']}")
            if 'analysis' in result:
                analysis = result['analysis']
                print(f"👥 People: {analysis.get('total_people', 0)}")
                if 'crowd_analysis' in analysis:
                    crowd = analysis['crowd_analysis']
                    print(f"🏃 Crowds: {crowd.get('total_crowds', 0)}")
            return result
        else:
            print(f"❌ Latest failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Latest error: {e}")
        return None

def test_cleanup():
    """Test cleanup old files"""
    try:
        # Test cleanup với 0 ngày (xóa tất cả file cũ)
        data = {"keep_days": 0}
        response = requests.post(f"{SERVER_URL}/cleanup", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n🗑️ CLEANUP RESULTS:")
            print(f"📁 Files deleted: {result.get('deleted_count', 0)}")
            print(f"💬 Message: {result.get('message', '')}")
            return result
        else:
            print(f"❌ Cleanup failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Cleanup error: {e}")
        return None

def test_clear_all():
    """Test xóa tất cả files"""
    try:
        response = requests.post(f"{SERVER_URL}/clear_all")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n🧹 CLEAR ALL RESULTS:")
            print(f"📁 Files deleted: {result.get('deleted_count', 0)}")
            print(f"💬 Message: {result.get('message', '')}")
            return result
        else:
            print(f"❌ Clear all failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Clear all error: {e}")
        return None

def main():
    print("🧪 TESTING AI CROWD DETECTION SERVER")
    print("=" * 50)
    
    # Test server stats
    test_get_stats()
    
    # Test với ảnh có sẵn
    test_images = []
    received_dir = "received_images"
    
    if os.path.exists(received_dir):
        image_files = [f for f in os.listdir(received_dir) if f.endswith('.jpg')]
        if image_files:
            # Lấy 2-3 ảnh đầu để test
            test_images = [os.path.join(received_dir, f) for f in image_files[:3]]
    
    if not test_images:
        print("❌ Không tìm thấy ảnh để test")
        print("💡 Hãy đặt một số ảnh .jpg vào thư mục received_images/")
        return
    
    print(f"\n🖼️  Testing với {len(test_images)} ảnh...")
    
    # Test upload từng ảnh
    for img_path in test_images:
        print(f"\n" + "="*30)
        test_upload_image(img_path)
        time.sleep(1)  # Chờ 1 giây giữa các request
    
    # Test lấy kết quả mới nhất
    test_get_latest()
    
    # Test cleanup functions
    print(f"\n" + "="*30)
    print("🧪 Testing cleanup functions...")
    test_cleanup()
    
    # Test stats cuối cùng
    print(f"\n" + "="*30)
    test_get_stats()
    
    print(f"\n✅ Test completed!")
    print(f"\n📋 Available endpoints:")
    print(f"   POST /upload - Upload ảnh và AI analysis")
    print(f"   GET /stats - Server statistics")
    print(f"   GET /latest - Latest analysis result")
    print(f"   GET /analysis/<filename> - Specific analysis")
    print(f"   POST /cleanup - Clean old files")
    print(f"   POST /clear_all - Delete all files")

if __name__ == "__main__":
    main()