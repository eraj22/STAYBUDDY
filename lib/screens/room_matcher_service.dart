import 'dart:convert';
import 'package:http/http.dart' as http;

/// Connects to the room_matcher Flask server.
/// Run locally:  cd room_matcher && python web_app.py
/// Default port: 5000
class RoomMatcherService {
  // Change this depending on where you run the app:
  // Web / desktop localhost:
  static const String baseUrl = 'http://127.0.0.1:5000';

  // Android emulator:
  // static const String baseUrl = 'http://10.0.2.2:5000';

  // Physical phone on same WiFi:
  // static const String baseUrl = 'http://192.168.1.10:5000';

  /// Find the best matching rooms for a resident.
  /// Uses /api/filter_rooms if preferences are given, else /api/find_rooms.
  static Future<List<Map<String, dynamic>>> findRooms({
    required String residentId,
    String? gender,
    String? roomType,
    String? residentType,
  }) async {
    final preferences = <String, dynamic>{};
    if (gender != null) preferences['gender'] = gender;
    if (roomType != null) preferences['room_type'] = roomType;
    if (residentType != null) preferences['resident_type'] = residentType;

    final uri = Uri.parse('$baseUrl/api/filter_rooms');
    final body = {
      'resident_id': residentId,
      'preferences': preferences,
    };

    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );

    if (response.statusCode != 200) {
      throw Exception('Room Matcher API error: ${response.body}');
    }

    final data = jsonDecode(response.body) as Map<String, dynamic>;
    if (data['error'] != null) {
      throw Exception(data['error']);
    }

    final rooms = data['rooms'] as List<dynamic>? ?? [];
    return rooms.cast<Map<String, dynamic>>();
  }

  /// Find best roommates for a resident.
  static Future<List<Map<String, dynamic>>> findRoommates({
    required String residentId,
    int topK = 5,
    String pool = 'all',
  }) async {
    final uri = Uri.parse('$baseUrl/api/find_roommates');
    final body = {
      'resident_id': residentId,
      'top_k': topK,
      'search_pool': pool,
    };

    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );

    if (response.statusCode != 200) {
      throw Exception('Roommate API error: ${response.body}');
    }

    final data = jsonDecode(response.body) as Map<String, dynamic>;
    if (data['error'] != null) throw Exception(data['error']);

    final roommates = data['roommates'] as List<dynamic>? ?? [];
    return roommates.cast<Map<String, dynamic>>();
  }

  /// Fetch a single resident's profile.
  static Future<Map<String, dynamic>> getProfile(String residentId) async {
    final uri = Uri.parse('$baseUrl/api/student/$residentId');
    final response = await http.get(uri);

    if (response.statusCode == 404) {
      throw Exception('Resident $residentId not found');
    }
    if (response.statusCode != 200) {
      throw Exception('Profile API error: ${response.body}');
    }

    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  /// Fetch all student profiles (for browse use).
  static Future<List<Map<String, dynamic>>> getAllStudents({
    String type = 'all',
    int limit = 50,
  }) async {
    final uri = Uri.parse(
        '$baseUrl/api/all_students?type=$type&limit=$limit');
    final response = await http.get(uri);

    if (response.statusCode != 200) {
      throw Exception('All students API error: ${response.body}');
    }

    final data = jsonDecode(response.body) as Map<String, dynamic>;
    final students = data['students'] as List<dynamic>? ?? [];
    return students.cast<Map<String, dynamic>>();
  }
}