from flask import Flask, request, jsonify
import os
from datetime import datetime
from rpi_crowd_detector import RPiCrowdDetector
import json
import numpy as np

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

app = Flask(__name__)
UPLOAD_FOLDER = "received_images"
RESULTS_FOLDER = "analysis_results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Khởi tạo AI detector
print("🤖 Đang khởi tạo AI Crowd Detector...")
try:
    ai_detector = RPiCrowdDetector(
        weights='yolov5s.pt',
        img_size=320,
        conf_thres=0.3
    )
    print("✅ AI Detector sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi khởi tạo AI Detector: {e}")
    ai_detector = None

@app.route('/upload', methods=['POST'])
def upload():
    # đổi từ "image" sang "file"
    if 'file' not in request.files:
        return {"status": "error", "message": "No file field"}, 400

    image = request.files['file']
    if image.filename == '':
        return {"status": "error", "message": "No file selected"}, 400

    # Lưu ảnh
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    print(f"✅ Received image: {image_path}")
    
    # Chạy AI phân tích đám đông
    analysis_result = None
    if ai_detector:
        try:
            print(f"🤖 Analyzing crowd in: {filename}")
            
            # Chạy AI detection
            result_path = os.path.join(RESULTS_FOLDER, f"analysis_{filename}")
            analysis_result = ai_detector.detect_single_image(
                image_path, 
                result_path, 
                visualize=True
            )
            
            if analysis_result:
                # Convert numpy types để tránh JSON serialization error
                analysis_result = convert_numpy_types(analysis_result)
                
                # Lưu kết quả JSON
                json_path = os.path.join(RESULTS_FOLDER, f"result_{filename}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result, f, indent=2, ensure_ascii=False)
                
                print(f"💾 Analysis saved: {json_path}")
                
        except Exception as e:
            print(f"❌ AI Analysis error: {e}")
            analysis_result = {"error": str(e)}
    
    # Xóa ảnh gốc sau khi xử lý xong để tiết kiệm dung lượng
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"🗑️ Deleted original image: {filename}")
    except Exception as e:
        print(f"⚠️ Could not delete image {filename}: {e}")
    
    # Tạo response với kết quả AI
    response_data = {
        "status": "success", 
        "filename": filename,
        "timestamp": datetime.now().isoformat(),
        "analysis": convert_numpy_types(analysis_result) if analysis_result else None,
        "note": "Original image deleted after processing"
    }
    
    # In summary nếu có kết quả
    if analysis_result and "crowd_analysis" in analysis_result:
        crowd_data = analysis_result["crowd_analysis"]
        print(f"📊 CROWD ANALYSIS SUMMARY:")
        print(f"   👥 Total people: {analysis_result.get('total_people', 0)}")
        print(f"   🏃 Crowds detected: {crowd_data.get('total_crowds', 0)}")
        print(f"   👫 People in crowds: {crowd_data.get('total_people_in_crowds', 0)}")
        print(f"   🚶 Isolated people: {crowd_data.get('isolated_people', 0)}")
    
    return jsonify(response_data), 200

@app.route('/analysis/<filename>', methods=['GET'])
def get_analysis(filename):
    """Lấy kết quả phân tích cho một ảnh cụ thể"""
    json_filename = f"result_{filename}.json"
    json_path = os.path.join(RESULTS_FOLDER, json_filename)
    
    if not os.path.exists(json_path):
        return {"status": "error", "message": "Analysis not found"}, 404
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        return jsonify({
            "status": "success",
            "filename": filename,
            "analysis": analysis_data
        }), 200
        
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.route('/latest', methods=['GET'])
def get_latest_analysis():
    """Lấy kết quả phân tích mới nhất"""
    try:
        # Tìm file JSON mới nhất
        json_files = [f for f in os.listdir(RESULTS_FOLDER) if f.endswith('.json')]
        if not json_files:
            return {"status": "error", "message": "No analysis found"}, 404
        
        latest_file = max(json_files, key=lambda f: os.path.getctime(os.path.join(RESULTS_FOLDER, f)))
        latest_path = os.path.join(RESULTS_FOLDER, latest_file)
        
        with open(latest_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        return jsonify({
            "status": "success",
            "filename": latest_file,
            "analysis": analysis_data
        }), 200
        
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Thống kê tổng quan"""
    try:
        # Đếm số ảnh đã nhận
        image_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.jpg')]
        
        # Đếm số phân tích đã thực hiện
        json_files = [f for f in os.listdir(RESULTS_FOLDER) if f.endswith('.json')]
        
        return jsonify({
            "status": "success",
            "stats": {
                "total_images_received": len(image_files),
                "total_analyses_completed": len(json_files),
                "ai_detector_status": "active" if ai_detector else "inactive"
            }
        }), 200
        
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.route('/cleanup', methods=['POST'])
def cleanup_old_files():
    """Dọn dẹp các file kết quả cũ"""
    try:
        # Lấy số ngày để giữ lại (mặc định 7 ngày)
        keep_days = request.json.get('keep_days', 7) if request.is_json else 7
        
        import time
        current_time = time.time()
        cutoff_time = current_time - (keep_days * 24 * 60 * 60)  # Convert days to seconds
        
        deleted_files = 0
        
        # Dọn dẹp thư mục received_images
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    file_time = os.path.getctime(file_path)
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        deleted_files += 1
                        print(f"🗑️ Deleted old image: {filename}")
        
        # Dọn dẹp thư mục analysis_results
        if os.path.exists(RESULTS_FOLDER):
            for filename in os.listdir(RESULTS_FOLDER):
                file_path = os.path.join(RESULTS_FOLDER, filename)
                if os.path.isfile(file_path):
                    file_time = os.path.getctime(file_path)
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        deleted_files += 1
                        print(f"🗑️ Deleted old result: {filename}")
        
        return jsonify({
            "status": "success",
            "message": f"Cleaned up {deleted_files} old files (older than {keep_days} days)",
            "deleted_count": deleted_files
        }), 200
        
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.route('/clear_all', methods=['POST'])
def clear_all_files():
    """Xóa tất cả file (chỉ dùng khi cần thiết)"""
    try:
        deleted_count = 0
        
        # Xóa tất cả ảnh
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_count += 1
        
        # Xóa tất cả kết quả
        if os.path.exists(RESULTS_FOLDER):
            for filename in os.listdir(RESULTS_FOLDER):
                file_path = os.path.join(RESULTS_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_count += 1
        
        print(f"🗑️ Cleared all files: {deleted_count} files deleted")
        
        return jsonify({
            "status": "success",
            "message": f"Cleared all files: {deleted_count} files deleted",
            "deleted_count": deleted_count
        }), 200
        
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

if __name__ == "__main__":
    # ⚠️ Quan trọng: host="0.0.0.0" để cho ESP32 truy cập qua LAN
    app.run(host="0.0.0.0", port=7861, debug=True)
