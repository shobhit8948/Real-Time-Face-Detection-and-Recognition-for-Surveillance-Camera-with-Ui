# this is app.py file

import cv2
from flask import Flask, Response, render_template, request, jsonify, redirect, url_for
from utils.database import  get_logs_for_person, initialize_database, get_all_persons, get_logs, delete_person_and_logs, delete_log
from utils.face_recognition import recognize_and_log, align_face
from utils.camera import start_camera, stop_camera, camera_stream
app = Flask(__name__)

# Initialize the database
initialize_database()

# Live feed status
camera_status = {"active": False}

@app.route('/')
def dashboard():
    stats = {
        "total_persons": len(get_all_persons()),
        "total_logs": len(get_logs()),
        "average_accuracy": "90%"  # Replace with dynamic computation if needed
    }
    return render_template('dashboard.html', stats=stats)

@app.route('/dashboard-data')
def dashboard_data():
    from datetime import datetime, timedelta

    filter = request.args.get("filter", "week")
    now = datetime.now()
    labels = []
    registered_counts = []
    visited_counts = []

    if filter == "week":
        for i in range(7):
            date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            labels.append(date)
            registered_counts.append(len(get_all_persons(filter_date=date)))
            visited_counts.append(len(get_logs(filter_date=date)))

    elif filter == "month":
        labels = ["Week 1", "Week 2", "Week 3", "Week 4"]
        for i in range(4):
            start_date = (now - timedelta(weeks=i+1)).strftime("%Y-%m-%d")
            end_date = (now - timedelta(weeks=i)).strftime("%Y-%m-%d")
            registered_counts.append(len(get_all_persons(start_date=start_date, end_date=end_date)))
            visited_counts.append(len(get_logs(start_date=start_date, end_date=end_date)))

    elif filter == "year":
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        for quarter, start_month in enumerate(range(1, 13, 3), start=1):
            start_date = now.replace(month=start_month, day=1).strftime("%Y-%m-%d")
            end_month = start_month + 2

            # Calculate the last day of the end month
            if end_month > 12:
                end_month -= 12
                end_year = now.year + 1
            else:
                end_year = now.year
            end_date = (now.replace(year=end_year, month=end_month, day=1) - timedelta(days=1)).strftime("%Y-%m-%d")

            labels.append(quarters[quarter - 1])
            registered_counts.append(len(get_all_persons(start_date=start_date, end_date=end_date)))
            visited_counts.append(len(get_logs(start_date=start_date, end_date=end_date)))

    return jsonify({
        "labels": labels,
        "registered": registered_counts,
        "visited": visited_counts
    })


@app.route('/person-management')
def person_management():
    filter_date = request.args.get('filter_date')
    if not filter_date:
        filter_date = datetime.now().strftime("%Y-%m-%d")
    persons = get_all_persons(filter_date)
    return render_template('person_management.html', persons=persons)

@app.route('/delete-person/<int:person_id>', methods=['POST'])
def delete_person(person_id):
    delete_person_and_logs(person_id)
    return redirect(url_for('person_management'))

@app.route('/person-details/<int:person_id>')
def person_details(person_id):
    filter_date = request.args.get('filter_date')  # Get filter_date from the request
    person = next((p for p in get_all_persons() if p["id"] == person_id), None)

    if not person:
        return "Person not found", 404

    # Fetch logs for the person filtered by date
    if filter_date:
        logs = [
            log for log in get_logs_for_person(person_id)
            if log["in_time"].startswith(filter_date)
        ]
    else:
        logs = get_logs_for_person(person_id)

    return render_template('person_details.html', person=person, logs=logs)

from datetime import datetime, timedelta

@app.route('/logs-management')
def logs_management():
    filter_date = request.args.get('filter_date')
    if not filter_date:
        filter_date = datetime.now().strftime("%Y-%m-%d")  # Default to the current date
    logs = get_logs(filter_date)
    return render_template('logs_management.html', logs=logs)


import os
from flask import send_from_directory

@app.route('/captured_faces/<path:filename>')
def serve_captured_face(filename):
    """Serve files from the captured_faces directory."""
    # directory = os.path.join(app.root_path, 'captured_faces')  # Adjust path
    return send_from_directory(app.root_path, filename)

@app.route('/delete-log/<int:log_id>', methods=['POST'])
def delete_log_route(log_id):  # Adjusted function name for clarity
    """Delete a log entry."""
    delete_log(log_id)  # Calls the database deletion function
    return redirect(url_for('logs_management'))

@app.route('/live-feed')
def live_feed():
    return render_template('live_feed.html', camera_running=camera_stream.running)


def generate_frames():
    """Generate processed video frames for streaming."""
    while camera_stream.running:
        frame = camera_stream.get_processed_frame()
        if frame is None:
            continue
        # Encode the processed frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video-feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle-feed', methods=['POST'])
def toggle_feed():
    if camera_stream.running:
        camera_stream.stop()
    else:
        camera_stream.start()
    return redirect(url_for('live_feed'))

@app.route('/search-logs', methods=['GET', 'POST'])
def search_logs():
    logs = []
    if request.method == 'POST':
        person_id = request.form.get('person_id')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        if person_id:
            all_logs = get_logs_for_person(int(person_id))
            if start_date and end_date:
                logs = [
                    log for log in all_logs
                    if start_date <= log["in_time"][:10] <= end_date
                ]
            else:
                logs = all_logs
        else:
            if start_date and end_date:
                logs = [
                    log for log in get_logs()
                    if start_date <= log["timestamp"][:10] <= end_date
                ]
            else:
                logs = get_logs()

    return render_template('search_logs.html', logs=logs)


if __name__ == '__main__':
    app.run(debug=True)
