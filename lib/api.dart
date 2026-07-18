import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'config.dart';

// ─── Auth Storage ─────────────────────────────────────────────────────────────
class AuthStore {
  static const _tokenKey = 'auth_token';
  static const _userKey = 'auth_user';

  static Future<void> saveToken(String token) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_tokenKey, token);
  }

  static Future<String?> getToken() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_tokenKey);
  }

  static Future<void> saveUser(Map<String, dynamic> user) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_userKey, jsonEncode(user));
  }

  static Future<Map<String, dynamic>?> getUser() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_userKey);
    if (raw == null) return null;
    return jsonDecode(raw) as Map<String, dynamic>;
  }

  static Future<void> clear() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_tokenKey);
    await prefs.remove(_userKey);
  }

  static Future<bool> isLoggedIn() async {
    final token = await getToken();
    return token != null && token.isNotEmpty;
  }

  static Future<String?> getRole() async {
    final user = await getUser();
    return user?['role'] as String?;
  }
}

// ─── API Client ───────────────────────────────────────────────────────────────
class Api {
  final String _base = AppConfig.baseUrl;

  Future<int> _currentUserId() async {
    final user = await AuthStore.getUser();
    final userId = int.tryParse(user?['id']?.toString() ?? '');
    if (userId == null || userId <= 0) {
      throw Exception('Please sign in to continue');
    }
    return userId;
  }

  Future<Map<String, String>> _authHeaders() async {
    final token = await AuthStore.getToken();
    return {
      'Content-Type': 'application/json',
      if (token != null) 'Authorization': 'Bearer $token',
    };
  }

  dynamic _parse(http.Response r, String label) {
    final body = r.body.trim();
    // Check if server returned HTML instead of JSON (e.g. 404 page, server not running)
    if (body.startsWith('<') || body.startsWith('<!')) {
      throw Exception(
        '$label failed [${r.statusCode}]: Server returned HTML — '
        'check that the Node.js backend is running on port 5000',
      );
    }
    if (r.statusCode >= 200 && r.statusCode < 300) {
      return jsonDecode(body);
    }
    try {
      final parsed = jsonDecode(body);
      throw Exception(
        '$label failed [${r.statusCode}]: ${parsed['error'] ?? body}',
      );
    } catch (e) {
      if (e is Exception) rethrow;
      throw Exception('$label failed [${r.statusCode}]: $body');
    }
  }

  // ══════════════════════════════════════════════════════════════════
  // AUTH
  // ══════════════════════════════════════════════════════════════════

  Future<Map<String, dynamic>> login(String email, String password) async {
    final r = await http.post(
      Uri.parse('$_base/api/auth/login'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'email': email, 'password': password}),
    );
    final data = _parse(r, 'Login') as Map<String, dynamic>;
    await AuthStore.saveToken(data['token'] as String);
    await AuthStore.saveUser(data['user'] as Map<String, dynamic>);
    return data;
  }

  Future<Map<String, dynamic>> register(
    String name,
    String email,
    String password, {
    String role = 'student',
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/auth/register'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'name': name,
        'email': email,
        'password': password,
        'role': role,
      }),
    );
    final data = _parse(r, 'Register') as Map<String, dynamic>;
    await AuthStore.saveToken(data['token'] as String);
    await AuthStore.saveUser(data['user'] as Map<String, dynamic>);
    return data;
  }

  Future<Map<String, dynamic>> registerOwner(
    String name,
    String email,
    String password,
  ) async {
    return register(name, email, password, role: 'owner');
  }

  Future<Map<String, dynamic>> getMe() async {
    final r = await http.get(
      Uri.parse('$_base/api/auth/me'),
      headers: await _authHeaders(),
    );
    return _parse(r, 'GetMe') as Map<String, dynamic>;
  }

  Future<void> logout() async {
    await AuthStore.clear();
  }

  // ══════════════════════════════════════════════════════════════════
  // STUDENT APIs
  // ══════════════════════════════════════════════════════════════════

  Future<List<dynamic>> getHostels({
    String? city,
    double? minRent,
    double? maxRent,
    List<String>? amenities,
    double? lat,
    double? lng,
    double? radiusKm,
  }) async {
    final params = <String, String>{};
    if (city != null) params['city'] = city;
    if (minRent != null) params['min_price'] = minRent.toString();
    if (maxRent != null) params['max_price'] = maxRent.toString();
    if (amenities != null && amenities.isNotEmpty)
      params['amenities'] = amenities.join(',');
    if (lat != null) params['latitude'] = lat.toString();
    if (lng != null) params['longitude'] = lng.toString();
    if (radiusKm != null) params['radius_km'] = radiusKm.toString();

    final endpoint = lat != null && lng != null
        ? '/api/hostels/search/nearby'
        : params.isEmpty
        ? '/api/hostels'
        : '/api/hostels/search/filter';
    final uri = Uri.parse('$_base$endpoint').replace(queryParameters: params);
    final r = await http.get(uri, headers: await _authHeaders());
    return _parse(r, 'GetHostels') as List<dynamic>;
  }

  Future<Map<String, dynamic>> getHostelDetail(int id) async {
    final r = await http.get(Uri.parse('$_base/api/hostels/$id'));
    return _parse(r, 'GetHostelDetail') as Map<String, dynamic>;
  }

  Future<List<dynamic>> getFavourites() async {
    final userId = await _currentUserId();
    final r = await http.get(
      Uri.parse('$_base/api/favorites/$userId'),
      headers: await _authHeaders(),
    );
    return _parse(r, 'GetFavourites') as List<dynamic>;
  }

  Future<Map<String, dynamic>> addFavourite(int hostelId) async {
    final r = await http.post(
      Uri.parse('$_base/api/favorites'),
      headers: await _authHeaders(),
      body: jsonEncode({'hostel_id': hostelId}),
    );
    return _parse(r, 'AddFavourite') as Map<String, dynamic>;
  }

  Future<void> removeFavourite(int hostelId) async {
    final userId = await _currentUserId();
    final r = await http.delete(
      Uri.parse('$_base/api/favorites/$userId/$hostelId'),
      headers: await _authHeaders(),
    );
    _parse(r, 'RemoveFavourite');
  }

  Future<List<dynamic>> getBookings() async {
    final r = await http.get(
      Uri.parse('$_base/api/bookings'),
      headers: await _authHeaders(),
    );
    return _parse(r, 'GetBookings') as List<dynamic>;
  }

  Future<Map<String, dynamic>> createBooking(
    int hostelId,
    String checkIn,
    String checkOut,
  ) async {
    final r = await http.post(
      Uri.parse('$_base/api/bookings'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'hostel_id': hostelId,
        'check_in': checkIn,
        'check_out': checkOut,
        'status': 'pending',
      }),
    );
    return _parse(r, 'CreateBooking') as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> cancelBooking(int bookingId) async {
    final r = await http.post(
      Uri.parse('$_base/api/bookings/$bookingId/cancel'),
      headers: await _authHeaders(),
    );
    return _parse(r, 'CancelBooking') as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> submitReview({
    required int hostelId,
    required double overallRating,
    double? cleanliness,
    double? facilities,
    double? management,
    String? textReview,
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/reviews'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'hostel_id': hostelId,
        'overall_rating': overallRating,
        if (cleanliness != null) 'cleanliness': cleanliness,
        if (facilities != null) 'facilities': facilities,
        if (management != null) 'management': management,
        if (textReview != null) 'text_review': textReview,
      }),
    );
    return _parse(r, 'SubmitReview') as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> fileComplaint({
    required int hostelId,
    required String category,
    required String severity,
    required String description,
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/complaints'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'hostel_id': hostelId,
        'category': category,
        'severity': severity,
        'description': description,
      }),
    );
    return _parse(r, 'FileComplaint') as Map<String, dynamic>;
  }

  Future<List<dynamic>> getComplaints() async {
    final r = await http.get(
      Uri.parse('$_base/api/complaints'),
      headers: await _authHeaders(),
    );
    return _parse(r, 'GetComplaints') as List<dynamic>;
  }

  Future<Map<String, dynamic>> getRecommendations({int topK = 5}) async {
    final uri = Uri.parse(
      '$_base/api/student/recommendations',
    ).replace(queryParameters: {'top_k': topK.toString()});
    final r = await http.get(uri, headers: await _authHeaders());
    return _parse(r, 'GetRecommendations') as Map<String, dynamic>;
  }

  // ══════════════════════════════════════════════════════════════════
  // OWNER APIs
  // ══════════════════════════════════════════════════════════════════

  /// Register a new hostel (called after owner completes all steps)
  Future<Map<String, dynamic>> registerHostel({
    required String name,
    required String hostelType,
    required String city,
    required String address,
    double? latitude,
    double? longitude,
    int? totalCapacity,
    int? floors,
    int? totalRooms,
    double? monthlyRent,
    List<String>? amenities,
    String? description,
    List<Map<String, dynamic>>? roomTypes,
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/owner/hostels'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'name': name,
        'hostel_type': hostelType,
        'city': city,
        'address': address,
        if (latitude != null) 'latitude': latitude,
        if (longitude != null) 'longitude': longitude,
        if (totalCapacity != null) 'total_capacity': totalCapacity,
        if (floors != null) 'floors': floors,
        if (totalRooms != null) 'total_rooms': totalRooms,
        if (monthlyRent != null) 'monthly_rent': monthlyRent,
        if (amenities != null) 'amenities': amenities,
        if (description != null) 'description': description,
        if (roomTypes != null) 'room_types': roomTypes,
      }),
    );
    return _parse(r, 'RegisterHostel') as Map<String, dynamic>;
  }

  Future<List<dynamic>> getOwnerHostels() async {
    final r = await http.get(
      Uri.parse('$_base/api/owner/hostels'),
      headers: await _authHeaders(),
    );
    return _parse(r, 'GetOwnerHostels') as List<dynamic>;
  }

  Future<Map<String, dynamic>> getOwnerDashboard() async {
    final r = await http.get(
      Uri.parse('$_base/api/owner/dashboard'),
      headers: await _authHeaders(),
    );
    return _parse(r, 'GetOwnerDashboard') as Map<String, dynamic>;
  }

  Future<List<dynamic>> getOwnerWardens() async {
    final r = await http.get(
      Uri.parse('$_base/api/owner/wardens'),
      headers: await _authHeaders(),
    );
    return _parse(r, 'GetOwnerWardens') as List<dynamic>;
  }

  Future<Map<String, dynamic>> addWarden({
    required String email,
    String? name,
    int? hostelId,
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/owner/wardens'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'email': email,
        if (name != null) 'name': name,
        if (hostelId != null) 'hostel_id': hostelId,
      }),
    );
    return _parse(r, 'AddWarden') as Map<String, dynamic>;
  }

  Future<List<dynamic>> getOwnerBookings() async {
    final r = await http.get(
      Uri.parse('$_base/api/owner/bookings'),
      headers: await _authHeaders(),
    );
    return _parse(r, 'GetOwnerBookings') as List<dynamic>;
  }

  Future<Map<String, dynamic>> updateOwnerBooking(
    int bookingId,
    String status,
  ) async {
    final r = await http.patch(
      Uri.parse('$_base/api/owner/bookings/$bookingId'),
      headers: await _authHeaders(),
      body: jsonEncode({'status': status}),
    );
    return _parse(r, 'UpdateOwnerBooking') as Map<String, dynamic>;
  }

  Future<List<dynamic>> getOwnerComplaints() async {
    final r = await http.get(
      Uri.parse('$_base/api/owner/complaints'),
      headers: await _authHeaders(),
    );
    return _parse(r, 'GetOwnerComplaints') as List<dynamic>;
  }

  // ══════════════════════════════════════════════════════════════════
  // WARDEN APIs
  // ══════════════════════════════════════════════════════════════════

  Future<Map<String, dynamic>> getWardenDashboard() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/dashboard'),
      headers: await _authHeaders(),
    );
    return _parse(r, 'GetWardenDashboard') as Map<String, dynamic>;
  }

  Future<List<dynamic>> getWardenComplaints() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/complaints'),
      headers: await _authHeaders(),
    );
    return _parse(r, 'GetWardenComplaints') as List<dynamic>;
  }

  Future<Map<String, dynamic>> updateWardenComplaint(
    int complaintId,
    String status,
  ) async {
    final r = await http.patch(
      Uri.parse('$_base/api/warden/complaints/$complaintId'),
      headers: await _authHeaders(),
      body: jsonEncode({'status': status}),
    );
    return _parse(r, 'UpdateWardenComplaint') as Map<String, dynamic>;
  }

  Future<List<dynamic>> getWardenBookings() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/bookings'),
      headers: await _authHeaders(),
    );
    return _parse(r, 'GetWardenBookings') as List<dynamic>;
  }

  Future<Map<String, dynamic>> updateWardenBooking(
    int bookingId,
    String status,
  ) async {
    final r = await http.patch(
      Uri.parse('$_base/api/warden/bookings/$bookingId'),
      headers: await _authHeaders(),
      body: jsonEncode({'status': status}),
    );
    return _parse(r, 'UpdateWardenBooking') as Map<String, dynamic>;
  }

  Future<List<dynamic>> getWardenAnnouncements() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/announcements'),
      headers: await _authHeaders(),
    );
    return _parse(r, 'GetWardenAnnouncements') as List<dynamic>;
  }

  Future<Map<String, dynamic>> postAnnouncement({
    required String title,
    required String body,
    String audience = 'All Students',
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/warden/announcements'),
      headers: await _authHeaders(),
      body: jsonEncode({'title': title, 'body': body, 'audience': audience}),
    );
    return _parse(r, 'PostAnnouncement') as Map<String, dynamic>;
  }

  Future<List<dynamic>> getWardenStudents() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/students'),
      headers: await _authHeaders(),
    );
    return _parse(r, 'GetWardenStudents') as List<dynamic>;
  }
}
