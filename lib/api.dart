import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import 'config.dart';

// =============================================================================
// AUTH STORAGE
// =============================================================================

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

    if (raw == null) {
      return null;
    }

    return Map<String, dynamic>.from(jsonDecode(raw) as Map);
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

    return user?['role']?.toString();
  }
}

// =============================================================================
// API
// =============================================================================

class Api {
  final String _base = AppConfig.baseUrl;

  Future<Map<String, String>> _authHeaders() async {
    final token = await AuthStore.getToken();

    return {
      'Content-Type': 'application/json',
      if (token != null && token.isNotEmpty) 'Authorization': 'Bearer $token',
    };
  }

  // ===========================================================================
  // AI ASSISTANT
  // ===========================================================================

  Future<List<dynamic>> getChatSessions({String? query, int offset = 0}) async {
    final uri = Uri.parse('$_base/api/chat/sessions').replace(
      queryParameters: {
        if (query != null && query.trim().isNotEmpty) 'q': query.trim(),
        'offset': offset.toString(),
      },
    );
    final response = await http.get(uri, headers: await _authHeaders());
    final data = Map<String, dynamic>.from(_parse(response, 'Get chats') as Map);
    return List<dynamic>.from(data['sessions'] as List? ?? const []);
  }

  Future<Map<String, dynamic>> createChatSession({String? title}) async {
    final response = await http.post(
      Uri.parse('$_base/api/chat/sessions'),
      headers: await _authHeaders(),
      body: jsonEncode({if (title != null) 'title': title}),
    );
    return Map<String, dynamic>.from(_parse(response, 'Create chat') as Map);
  }

  Future<List<dynamic>> getChatMessages(String sessionId) async {
    final response = await http.get(
      Uri.parse('$_base/api/chat/sessions/$sessionId/messages'),
      headers: await _authHeaders(),
    );
    final data = Map<String, dynamic>.from(_parse(response, 'Get messages') as Map);
    return List<dynamic>.from(data['messages'] as List? ?? const []);
  }

  Future<Map<String, dynamic>> renameChatSession(String sessionId, String title) async {
    final response = await http.patch(
      Uri.parse('$_base/api/chat/sessions/$sessionId'),
      headers: await _authHeaders(),
      body: jsonEncode({'title': title.trim()}),
    );
    return Map<String, dynamic>.from(_parse(response, 'Rename chat') as Map);
  }

  Future<void> deleteChatSession(String sessionId) async {
    final response = await http.delete(
      Uri.parse('$_base/api/chat/sessions/$sessionId'),
      headers: await _authHeaders(),
    );
    _parse(response, 'Delete chat');
  }

  Future<Map<String, dynamic>> sendChatMessage({
    required String sessionId,
    required String message,
    String provider = 'groq',
    String language = 'en',
    String? requestId,
    http.Client? client,
  }) async {
    final requester = client ?? http.Client();
    try {
      final response = await requester.post(
        Uri.parse('$_base/api/chat/sessions/$sessionId/messages'),
        headers: await _authHeaders(),
        body: jsonEncode({
          'message': message.trim(),
          'provider': provider,
          'language': language,
          'request_id': requestId ?? DateTime.now().microsecondsSinceEpoch.toString(),
        }),
      );
      return Map<String, dynamic>.from(_parse(response, 'Send message') as Map);
    } finally {
      if (client == null) requester.close();
    }
  }

  Future<Map<String, dynamic>> submitChatFeedback(int messageId, {required bool helpful}) async {
    final response = await http.post(
      Uri.parse('$_base/api/chat/messages/$messageId/feedback'),
      headers: await _authHeaders(),
      body: jsonEncode({'helpful': helpful}),
    );
    return Map<String, dynamic>.from(_parse(response, 'Save feedback') as Map);
  }

  Future<Map<String, dynamic>> executeAssistantAction(Map<String, dynamic> action) async {
    final path = action['path']?.toString();
    final method = action['method']?.toString().toUpperCase() ?? 'POST';
    if (path == null || !path.startsWith('/api/')) {
      throw Exception('This assistant action is not valid.');
    }
    final body = action['body'] is Map
        ? Map<String, dynamic>.from(action['body'] as Map)
        : <String, dynamic>{};
    final headers = await _authHeaders();
    final uri = Uri.parse('$_base$path');
    late final http.Response response;
    switch (method) {
      case 'PATCH':
        response = await http.patch(uri, headers: headers, body: jsonEncode(body));
        break;
      case 'PUT':
        response = await http.put(uri, headers: headers, body: jsonEncode(body));
        break;
      case 'DELETE':
        response = await http.delete(uri, headers: headers, body: jsonEncode(body));
        break;
      default:
        response = await http.post(uri, headers: headers, body: jsonEncode(body));
    }
    final value = _parse(response, 'Assistant action');
    return value is Map ? Map<String, dynamic>.from(value) : <String, dynamic>{};
  }

  dynamic _parse(http.Response response, String label) {
    final body = response.body.trim();

    if (body.isEmpty) {
      if (response.statusCode >= 200 && response.statusCode < 300) {
        return <String, dynamic>{};
      }

      throw Exception(
        '$label failed [${response.statusCode}]: Empty server response',
      );
    }

    if (body.startsWith('<') || body.startsWith('<!')) {
      throw Exception(
        '$label failed [${response.statusCode}]: '
        'Server returned HTML. Check that the Node.js backend is running on port 5000.',
      );
    }

    if (response.statusCode >= 200 && response.statusCode < 300) {
      return jsonDecode(body);
    }

    try {
      final parsed = jsonDecode(body);

      if (parsed is Map) {
        throw Exception(
          '$label failed [${response.statusCode}]: '
          '${parsed['error'] ?? parsed['message'] ?? body}',
        );
      }

      throw Exception('$label failed [${response.statusCode}]: $body');
    } catch (error) {
      if (error is Exception) {
        rethrow;
      }

      throw Exception('$label failed [${response.statusCode}]: $body');
    }
  }

  // ===========================================================================
  // AUTH
  // ===========================================================================

  Future<Map<String, dynamic>> login(String email, String password) async {
    final response = await http
        .post(
          Uri.parse('$_base/api/auth/login'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'email': email.trim().toLowerCase(),
            'password': password,
          }),
        )
        .timeout(const Duration(seconds: 10));

    final data = Map<String, dynamic>.from(_parse(response, 'Login') as Map);

    final token = data['token']?.toString();

    final userRaw = data['user'];

    if (token != null && token.isNotEmpty) {
      await AuthStore.saveToken(token);
    }

    if (userRaw is Map) {
      await AuthStore.saveUser(Map<String, dynamic>.from(userRaw));
    }

    return data;
  }

  Future<Map<String, dynamic>> register(
    String name,
    String email,
    String password, {
    String role = 'student',
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/auth/register'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'name': name.trim(),
        'email': email.trim().toLowerCase(),
        'password': password,
        'role': role,
      }),
    );

    final data = Map<String, dynamic>.from(_parse(response, 'Register') as Map);

    final token = data['token']?.toString();

    final userRaw = data['user'];

    if (token != null && token.isNotEmpty) {
      await AuthStore.saveToken(token);
    }

    if (userRaw is Map) {
      await AuthStore.saveUser(Map<String, dynamic>.from(userRaw));
    }

    return data;
  }

  Future<Map<String, dynamic>> requestPasswordReset(String email) async {
    final response = await http.post(
      Uri.parse('$_base/api/auth/password-reset/request'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'email': email.trim().toLowerCase()}),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'RequestPasswordReset') as Map,
    );
  }

  // Do NOT use public register endpoint for Owner creation.
  Future<Map<String, dynamic>> getMe() async {
    final response = await http.get(
      Uri.parse('$_base/api/auth/me'),
      headers: await _authHeaders(),
    );

    return Map<String, dynamic>.from(_parse(response, 'GetMe') as Map);
  }

  /// Returns only confirmed, room-assigned hostel memberships for the
  /// signed-in student. Empty assignments means discovery-only access.
  Future<Map<String, dynamic>> getStudentAccess() async {
    final response = await http.get(
      Uri.parse('$_base/api/student/access'),
      headers: await _authHeaders(),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'GetStudentAccess') as Map,
    );
  }

  Future<Map<String, dynamic>> getStudentProfile() async {
    final response = await http.get(
      Uri.parse('$_base/api/student/profile'),
      headers: await _authHeaders(),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'GetStudentProfile') as Map,
    );
  }

  Future<Map<String, dynamic>> updateStudentProfile(
    Map<String, dynamic> values,
  ) async {
    final response = await http.put(
      Uri.parse('$_base/api/student/profile'),
      headers: await _authHeaders(),
      body: jsonEncode(values),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'UpdateStudentProfile') as Map,
    );
  }

  Future<Map<String, dynamic>> getBedAvailability(int hostelId) async {
    final response = await http.get(
      Uri.parse('$_base/api/student/hostels/$hostelId/bed-availability'),
      headers: await _authHeaders(),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'GetBedAvailability') as Map,
    );
  }

  Future<Map<String, dynamic>> createRoomChangeRequest({
    required int hostelId,
    required String reason,
    required int requestedRoomId,
    required String requestedBedNumber,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/student/hostels/$hostelId/room-change-requests'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'reason': reason.trim(),
        'requested_room_id': requestedRoomId,
        'requested_bed_number': requestedBedNumber.trim(),
      }),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'CreateRoomChangeRequest') as Map,
    );
  }

  Future<List<dynamic>> getStudentRoomChangeRequests() async {
    final response = await http.get(
      Uri.parse('$_base/api/student/room-change-requests'),
      headers: await _authHeaders(),
    );
    final data = Map<String, dynamic>.from(
      _parse(response, 'GetStudentRoomChangeRequests') as Map,
    );
    return List<dynamic>.from(data['requests'] as List? ?? const []);
  }

  Future<Map<String, dynamic>> submitHostelFeedback({
    required int hostelId,
    required String type,
    required int rating,
    String? comment,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/student/hostels/$hostelId/feedback'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'type': type,
        'rating': rating,
        'comment': comment?.trim() ?? '',
      }),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'SubmitHostelFeedback') as Map,
    );
  }

  Future<Map<String, dynamic>> getRoomPreferences(int hostelId) async {
    final response = await http.get(
      Uri.parse('$_base/api/student/hostels/$hostelId/preferences'),
      headers: await _authHeaders(),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'GetRoomPreferences') as Map,
    );
  }

  Future<Map<String, dynamic>> saveRoomPreferences({
    required int hostelId,
    String? culture,
    String? caste,
    String? university,
  }) async {
    final response = await http.put(
      Uri.parse('$_base/api/student/hostels/$hostelId/preferences'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'culture_preference': culture?.trim() ?? '',
        'caste_preference': caste?.trim() ?? '',
        'university_preference': university?.trim() ?? '',
      }),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'SaveRoomPreferences') as Map,
    );
  }

  Future<Map<String, dynamic>> getHostelChat({
    required int hostelId,
    required String channel,
  }) async {
    final response = await http.get(
      Uri.parse('$_base/api/student/hostels/$hostelId/chat/$channel'),
      headers: await _authHeaders(),
    );
    return Map<String, dynamic>.from(_parse(response, 'GetHostelChat') as Map);
  }

  Future<Map<String, dynamic>> sendHostelChat({
    required int hostelId,
    required String channel,
    required String body,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/student/hostels/$hostelId/chat/$channel'),
      headers: await _authHeaders(),
      body: jsonEncode({'body': body.trim()}),
    );
    return Map<String, dynamic>.from(_parse(response, 'SendHostelChat') as Map);
  }

  Future<List<dynamic>> getAnnouncements() async {
    final response = await http.get(
      Uri.parse('$_base/api/announcements'),
      headers: await _authHeaders(),
    );
    return _parse(response, 'GetAnnouncements') as List<dynamic>;
  }

  /// Student's own verified invoice and balance statement.  The server
  /// calculates payments from approved receipts and warden cash entries.
  Future<Map<String, dynamic>> getStudentFeeStatement() async {
    final response = await http.get(
      Uri.parse('$_base/api/fee-statements'),
      headers: await _authHeaders(),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'GetStudentFeeStatement') as Map,
    );
  }

  Future<Map<String, dynamic>> getInvoicePaymentDestination(
    int invoiceId,
  ) async {
    final response = await http.get(
      Uri.parse('$_base/api/fee-invoices/$invoiceId/payment-destination'),
      headers: await _authHeaders(),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'GetInvoicePaymentDestination') as Map,
    );
  }

  Future<Map<String, dynamic>> submitFeeReceipt({
    required int invoiceId,
    required int paymentDestinationId,
    required double amount,
    required String paymentReference,
    required DateTime paidAt,
    required String filename,
    required String contentType,
    required String contentBase64,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/fee-invoices/$invoiceId/receipt-submissions'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'payment_destination_id': paymentDestinationId,
        'amount': amount,
        'payment_reference': paymentReference.trim(),
        'paid_at': paidAt.toIso8601String(),
        'filename': filename,
        'content_type': contentType,
        'content_base64': contentBase64,
      }),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'SubmitFeeReceipt') as Map,
    );
  }

  Future<Map<String, dynamic>> submitStudentPaymentProof({
    required int invoiceId,
    required double amount,
    required String paymentReference,
    required DateTime paidAt,
    required String filename,
    required String contentType,
    required String contentBase64,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/fee-invoices/$invoiceId/payment-proofs'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'amount': amount,
        'payment_reference': paymentReference.trim(),
        'paid_at': paidAt.toIso8601String(),
        'filename': filename,
        'content_type': contentType,
        'content_base64': contentBase64,
      }),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'SubmitStudentPaymentProof') as Map,
    );
  }

  Future<void> logout() async {
    await AuthStore.clear();
  }

  // ===========================================================================
  // OWNER REGISTRATION / OTP
  // ===========================================================================

  Future<Map<String, dynamic>> sendOwnerOtp({
    required String name,
    required String email,
    required String password,
    required String cnic,
    required String city,
    required String phone,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/owner/send-otp'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'full_name': name.trim(),
        'email': email.trim().toLowerCase(),
        'password': password,
        'cnic': cnic.trim(),
        'city': city.trim(),
        'phone': phone.trim(),
      }),
    );

    return Map<String, dynamic>.from(_parse(response, 'SendOwnerOtp') as Map);
  }

  Future<Map<String, dynamic>> verifyOwnerOtp({
    required String email,
    required String otp,
    required String password,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/owner/verify-otp'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'email': email.trim().toLowerCase(),
        'otp': otp.trim(),
        'password': password,
      }),
    );

    return Map<String, dynamic>.from(_parse(response, 'VerifyOwnerOtp') as Map);
  }

  // ===========================================================================
  // STUDENT APIs
  // ===========================================================================

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

    if (city != null) {
      params['city'] = city;
    }

    if (minRent != null) {
      params['min_rent'] = minRent.toString();
    }

    if (maxRent != null) {
      params['max_rent'] = maxRent.toString();
    }

    if (amenities != null && amenities.isNotEmpty) {
      params['amenities'] = amenities.join(',');
    }

    if (lat != null) {
      params['lat'] = lat.toString();
    }

    if (lng != null) {
      params['lng'] = lng.toString();
    }

    if (radiusKm != null) {
      params['radius_km'] = radiusKm.toString();
    }

    final uri = Uri.parse(
      '$_base/api/student/hostels',
    ).replace(queryParameters: params);

    final response = await http.get(uri, headers: await _authHeaders());

    return _parse(response, 'GetHostels') as List<dynamic>;
  }

  Future<Map<String, dynamic>> getHostelDetail(int id) async {
    final response = await http.get(
      Uri.parse('$_base/api/hostels/$id'),
      headers: await _authHeaders(),
    );

    return Map<String, dynamic>.from(
      _parse(response, 'GetHostelDetail') as Map,
    );
  }

  Future<List<dynamic>> getFavourites() async {
    final response = await http.get(
      Uri.parse('$_base/api/student/favourites'),
      headers: await _authHeaders(),
    );

    return _parse(response, 'GetFavourites') as List<dynamic>;
  }

  Future<Map<String, dynamic>> addFavourite(int hostelId) async {
    final response = await http.post(
      Uri.parse('$_base/api/student/favourites'),
      headers: await _authHeaders(),
      body: jsonEncode({'hostel_id': hostelId}),
    );

    return Map<String, dynamic>.from(_parse(response, 'AddFavourite') as Map);
  }

  Future<void> removeFavourite(int hostelId) async {
    final response = await http.delete(
      Uri.parse('$_base/api/student/favourites/$hostelId'),
      headers: await _authHeaders(),
    );

    _parse(response, 'RemoveFavourite');
  }

  Future<List<dynamic>> getBookings() async {
    final response = await http.get(
      Uri.parse('$_base/api/bookings'),
      headers: await _authHeaders(),
    );

    return _parse(response, 'GetBookings') as List<dynamic>;
  }

  Future<Map<String, dynamic>> createBooking(
    int hostelId,
    String checkIn,
    String checkOut, {
    String? requestedRoomType,
    Map<String, dynamic>? application,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/bookings'),
      headers: {
        ...await _authHeaders(),
        'Idempotency-Key': 'booking-${DateTime.now().microsecondsSinceEpoch}',
      },
      body: jsonEncode({
        'hostel_id': hostelId,
        'check_in': checkIn,
        'check_out': checkOut,
        if (requestedRoomType != null && requestedRoomType.trim().isNotEmpty)
          'requested_room_type': requestedRoomType.trim().toLowerCase(),
        if (application != null) 'application': application,
      }),
    );

    return Map<String, dynamic>.from(_parse(response, 'CreateBooking') as Map);
  }

  Future<List<dynamic>> getStudentHostelRooms(int hostelId) async {
    final response = await http.get(
      Uri.parse('$_base/api/student/hostels/$hostelId/rooms'),
      headers: await _authHeaders(),
    );
    return _parse(response, 'GetStudentHostelRooms') as List<dynamic>;
  }

  Future<List<dynamic>> getStudentRoomMatches(int bookingId) async {
    final response = await http.get(
      Uri.parse('$_base/api/student/bookings/$bookingId/room-matches'),
      headers: await _authHeaders(),
    );
    final data = Map<String, dynamic>.from(
      _parse(response, 'GetStudentRoomMatches') as Map,
    );
    return List<dynamic>.from(data['matches'] as List? ?? const []);
  }

  Future<Map<String, dynamic>> activateHostelAccess({
    required int bookingId,
    required String password,
  }) async {
    final response = await http.post(
      Uri.parse(
        '$_base/api/student/bookings/$bookingId/hostel-access/activate',
      ),
      headers: await _authHeaders(),
      body: jsonEncode({'password': password}),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'ActivateHostelAccess') as Map,
    );
  }

  Future<Map<String, dynamic>> selectStudentBed({
    required int bookingId,
    required int roomId,
    required String bedNumber,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/student/bookings/$bookingId/select-bed'),
      headers: await _authHeaders(),
      body: jsonEncode({'room_id': roomId, 'bed_number': bedNumber}),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'SelectStudentBed') as Map,
    );
  }

  Future<Map<String, dynamic>> submitReview({
    required int hostelId,
    required double overallRating,
    double? cleanliness,
    double? facilities,
    double? management,
    String? textReview,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/student/reviews'),
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

    return Map<String, dynamic>.from(_parse(response, 'SubmitReview') as Map);
  }

  Future<Map<String, dynamic>> fileComplaint({
    required int hostelId,
    required String category,
    required String severity,
    required String description,
    List<String> attachments = const [],
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/complaints'),
      headers: {
        ...await _authHeaders(),
        'Idempotency-Key': 'complaint-${DateTime.now().microsecondsSinceEpoch}',
      },
      body: jsonEncode({
        'hostel_id': hostelId,
        'category': category,
        'severity': severity,
        'description': description,
        'attachments': attachments,
      }),
    );

    return Map<String, dynamic>.from(_parse(response, 'FileComplaint') as Map);
  }

  Future<List<dynamic>> getComplaints() async {
    final response = await http.get(
      Uri.parse('$_base/api/complaints'),
      headers: await _authHeaders(),
    );

    return _parse(response, 'GetComplaints') as List<dynamic>;
  }

  Future<Map<String, dynamic>> getRecommendations({int topK = 5}) async {
    final uri = Uri.parse(
      '$_base/api/student/recommendations',
    ).replace(queryParameters: {'top_k': topK.toString()});

    final response = await http.get(uri, headers: await _authHeaders());

    return Map<String, dynamic>.from(
      _parse(response, 'GetRecommendations') as Map,
    );
  }

  Future<Map<String, dynamic>> getLiveMlRecommendations({
    int topK = 5,
    int? budgetMax,
    List<String> requiredAmenities = const [],
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/recommendations/ml'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'top_k': topK,
        if (budgetMax != null) 'budget_max': budgetMax,
        'required_amenities': requiredAmenities,
      }),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'GetLiveMlRecommendations') as Map,
    );
  }

  // ===========================================================================
  // OWNER APIs
  // ===========================================================================

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
    List<Map<String, double>>? attendanceBoundary,
    List<Map<String, dynamic>>? rooms,
    List<String>? facilities,
    List<String>? rules,
    Map<String, dynamic>? messInfo,
    List<Map<String, dynamic>>? media,
    String? contactPhone,
    String? contactEmail,
  }) async {
    final response = await http.post(
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
        if (attendanceBoundary != null)
          'attendance_boundary': attendanceBoundary,
        if (rooms != null) 'rooms': rooms,
        if (facilities != null) 'facilities': facilities,
        if (rules != null) 'rules': rules,
        if (messInfo != null) 'mess_info': messInfo,
        if (media != null) 'media': media,
        if (contactPhone != null) 'contact_phone': contactPhone.trim(),
        if (contactEmail != null) 'contact_email': contactEmail.trim(),
      }),
    );

    return Map<String, dynamic>.from(_parse(response, 'RegisterHostel') as Map);
  }

  Future<List<dynamic>> getOwnerHostels() async {
    final response = await http.get(
      Uri.parse('$_base/api/owner/hostels'),
      headers: await _authHeaders(),
    );

    return _parse(response, 'GetOwnerHostels') as List<dynamic>;
  }

  Future<Map<String, dynamic>> updateOwnerHostelAttendanceBoundary({
    required int hostelId,
    required List<Map<String, double>> boundary,
  }) async {
    final response = await http.put(
      Uri.parse('$_base/api/owner/hostels/$hostelId/attendance-geofence'),
      headers: await _authHeaders(),
      body: jsonEncode({'boundary': boundary}),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'UpdateOwnerHostelAttendanceBoundary') as Map,
    );
  }

  Future<Map<String, dynamic>> getOwnerDashboard() async {
    final response = await http.get(
      Uri.parse('$_base/api/owner/dashboard'),
      headers: await _authHeaders(),
    );

    return Map<String, dynamic>.from(
      _parse(response, 'GetOwnerDashboard') as Map,
    );
  }

  Future<Map<String, dynamic>> getOwnerReports() async {
    final response = await http.get(
      Uri.parse('$_base/api/owner/reports'),
      headers: await _authHeaders(),
    );

    return Map<String, dynamic>.from(
      _parse(response, 'GetOwnerReports') as Map,
    );
  }

  Future<List<dynamic>> getOwnerFeeRecords(String billingMonth) async {
    final uri = Uri.parse(
      '$_base/api/owner/fee-records',
    ).replace(queryParameters: {'billing_month': billingMonth});
    final response = await http.get(uri, headers: await _authHeaders());
    final data = Map<String, dynamic>.from(
      _parse(response, 'GetOwnerFeeRecords') as Map,
    );
    return List<dynamic>.from(data['records'] as List? ?? const []);
  }

  Future<void> setupOwnerRentRecord({
    required int bookingId,
    required String billingMonth,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/owner/fee-records/$bookingId/setup'),
      headers: await _authHeaders(),
      body: jsonEncode({'billing_month': billingMonth}),
    );
    _parse(response, 'SetupOwnerRentRecord');
  }

  Future<void> recordOwnerCashFeePayment({
    required int invoiceId,
    required double amount,
    required String receivedOn,
    String? note,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/owner/cash-fee-payments'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'invoice_id': invoiceId,
        'amount': amount,
        'received_on': receivedOn,
        if (note != null && note.trim().isNotEmpty) 'note': note.trim(),
      }),
    );
    _parse(response, 'RecordOwnerCashFeePayment');
  }

  Future<List<dynamic>> getHostelRooms(int hostelId) async {
    final response = await http.get(
      Uri.parse('$_base/api/hostels/$hostelId/rooms'),
      headers: await _authHeaders(),
    );

    return _parse(response, 'GetHostelRooms') as List<dynamic>;
  }

  Future<Map<String, dynamic>> addRoom({
    required int hostelId,
    required String roomNumber,
    required String floorLabel,
    required int capacity,
    required String roomType,
    required int monthlyRent,
    required List<String> bedNumbers,
    required List<String> amenities,
    String? notes,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/rooms'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'hostel_id': hostelId,
        'room_number': roomNumber.trim(),
        'floor_label': floorLabel.trim(),
        'capacity': capacity,
        'room_type': roomType.trim(),
        'monthly_rent': monthlyRent,
        'bed_numbers': bedNumbers,
        'amenities': amenities,
        if (notes != null && notes.trim().isNotEmpty) 'notes': notes.trim(),
      }),
    );

    return Map<String, dynamic>.from(_parse(response, 'AddRoom'));
  }

  Future<Map<String, dynamic>> updateRoom({
    required int roomId,
    required String roomNumber,
    String? floorLabel,
    required int capacity,
    required int availableCapacity,
    required String roomType,
    required int monthlyRent,
    List<String>? amenities,
    String? notes,
  }) async {
    final response = await http.patch(
      Uri.parse('$_base/api/rooms/$roomId'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'room_number': roomNumber.trim(),
        if (floorLabel != null) 'floor_label': floorLabel.trim(),
        'capacity': capacity,
        'available_capacity': availableCapacity,
        'room_type': roomType.trim(),
        'monthly_rent': monthlyRent,
        if (amenities != null) 'amenities': amenities,
        if (notes != null) 'notes': notes.trim(),
      }),
    );

    return Map<String, dynamic>.from(_parse(response, 'UpdateRoom') as Map);
  }

  Future<Map<String, dynamic>> updateOwnerHostel({
    required int hostelId,
    required Map<String, dynamic> updates,
  }) async {
    final response = await http.patch(
      Uri.parse('$_base/api/owner/hostels/$hostelId'),
      headers: await _authHeaders(),
      body: jsonEncode(updates),
    );

    return Map<String, dynamic>.from(
      _parse(response, 'UpdateOwnerHostel') as Map,
    );
  }

  Future<List<dynamic>> getOwnerWardens() async {
    final response = await http.get(
      Uri.parse('$_base/api/owner/wardens'),
      headers: await _authHeaders(),
    );

    return _parse(response, 'GetOwnerWardens') as List<dynamic>;
  }

  Future<Map<String, dynamic>> updateWardenAccountStatus({
    required int wardenId,
    required bool enabled,
  }) async {
    final response = await http.patch(
      Uri.parse('$_base/api/owner/wardens/$wardenId/account-status'),
      headers: await _authHeaders(),
      body: jsonEncode({'status': enabled ? 'active' : 'disabled'}),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'UpdateWardenAccountStatus') as Map,
    );
  }

  Future<Map<String, dynamic>> addWarden({
    required int hostelId,
    required String name,
    required String email,
    required String phone,
    required String cnic,
    required String city,
    required String shift,
    required List<String> floors,
    required Map<String, bool> permissions,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/owner/wardens'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'hostel_id': hostelId,
        'name': name.trim(),
        'email': email.trim().toLowerCase(),
        'phone': phone.trim(),
        'cnic': cnic.trim(),
        'city': city.trim(),
        'shift': shift,
        'floors': floors,
        'permissions': permissions,
      }),
    );

    return Map<String, dynamic>.from(_parse(response, 'AddWarden') as Map);
  }

  Future<List<dynamic>> getOwnerStudents() async {
    final response = await http.get(
      Uri.parse('$_base/api/owner/students'),
      headers: await _authHeaders(),
    );

    final data = _parse(response, 'GetOwnerStudents');

    if (data is Map && data['students'] is List) {
      return List<dynamic>.from(data['students']);
    }

    return [];
  }

  Future<Map<String, dynamic>> addOwnerStudent({
    required String name,
    required String email,
    required String phone,
    String? cnic,
    String? city,
    String? university,
    String? department,
    String? semester,
    String? parentName,
    String? parentPhone,
    String? parentEmail,
    String? parentRelation,
    String? parentCity,
    int? hostelId,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/owner/students'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'name': name.trim(),
        'email': email.trim().toLowerCase(),
        'phone': phone.trim(),
        if (cnic != null) 'cnic': cnic.trim(),
        if (city != null) 'city': city.trim(),
        if (university != null) 'university': university.trim(),
        if (department != null) 'department': department.trim(),
        if (semester != null) 'semester': semester.trim(),
        if (parentName != null) 'parent_name': parentName.trim(),
        if (parentPhone != null) 'parent_phone': parentPhone.trim(),
        if (parentEmail != null) 'parent_email': parentEmail.trim(),
        if (parentRelation != null) 'parent_relation': parentRelation.trim(),
        if (parentCity != null) 'parent_city': parentCity.trim(),
        if (hostelId != null) 'hostel_id': hostelId,
      }),
    );

    return Map<String, dynamic>.from(
      _parse(response, 'AddOwnerStudent') as Map,
    );
  }

  Future<Map<String, dynamic>> updateOwnerStudent({
    required int studentId,
    required Map<String, dynamic> updates,
  }) async {
    final response = await http.patch(
      Uri.parse('$_base/api/owner/students/$studentId'),
      headers: await _authHeaders(),
      body: jsonEncode(updates),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'UpdateOwnerStudent') as Map,
    );
  }

  Future<Map<String, dynamic>> setOwnerUserAccountStatus(int userId, bool enabled) async {
    final response = await http.patch(Uri.parse('$_base/api/owner/users/$userId/account-status'), headers: await _authHeaders(), body: jsonEncode({'status': enabled ? 'active' : 'disabled'}));
    return Map<String, dynamic>.from(_parse(response, 'SetOwnerUserAccountStatus') as Map);
  }

  Future<List<dynamic>> resetOwnerUserPasswords(List<int> userIds) async {
    final response = await http.post(Uri.parse('$_base/api/owner/users/password-reset'), headers: await _authHeaders(), body: jsonEncode({'user_ids': userIds}));
    final data = Map<String, dynamic>.from(_parse(response, 'ResetOwnerUserPasswords') as Map);
    return List<dynamic>.from(data['resets'] as List? ?? const []);
  }

  Future<void> revokeOwnerUserSessions(int userId) async {
    final response = await http.post(Uri.parse('$_base/api/owner/users/$userId/logout'), headers: await _authHeaders());
    _parse(response, 'RevokeOwnerUserSessions');
  }

  Future<List<dynamic>> getOwnerAuditEvents() async {
    final response = await http.get(Uri.parse('$_base/api/owner/audit-events'), headers: await _authHeaders());
    final data = Map<String, dynamic>.from(_parse(response, 'GetOwnerAuditEvents') as Map);
    return List<dynamic>.from(data['events'] as List? ?? const []);
  }

  Future<void> revokeOtherOwnerSessions() async {
    final response = await http.post(
      Uri.parse('$_base/api/owner/sessions/revoke-other-devices'),
      headers: await _authHeaders(),
    );
    _parse(response, 'Revoke other owner sessions');
  }

  Future<List<dynamic>> getOwnerSessions() async {
    final response = await http.get(Uri.parse('$_base/api/owner/sessions'), headers: await _authHeaders());
    final data = Map<String, dynamic>.from(_parse(response, 'Get owner sessions') as Map);
    return List<dynamic>.from(data['sessions'] as List? ?? const []);
  }

  Future<void> revokeOwnerSession(String sessionId) async {
    final response = await http.delete(Uri.parse('$_base/api/owner/sessions/$sessionId'), headers: await _authHeaders());
    _parse(response, 'Revoke owner session');
  }

  Future<List<dynamic>> getOwnerAvailableRooms(int bookingId) async {
    final response = await http.get(Uri.parse('$_base/api/owner/bookings/$bookingId/available-rooms'), headers: await _authHeaders());
    final data = Map<String, dynamic>.from(_parse(response, 'Get available rooms') as Map);
    return List<dynamic>.from(data['rooms'] as List? ?? const []);
  }

  Future<Map<String, dynamic>> moveOwnerStudentRoom(int bookingId, int roomId) async {
    final response = await http.post(Uri.parse('$_base/api/owner/bookings/$bookingId/move-room'), headers: await _authHeaders(), body: jsonEncode({'room_id': roomId}));
    return Map<String, dynamic>.from(_parse(response, 'Move student room') as Map);
  }

  Future<Map<String, dynamic>> getOwnerHostelRules(int hostelId) async {
    final response = await http.get(
      Uri.parse('$_base/api/owner/hostels/$hostelId/rules'),
      headers: await _authHeaders(),
    );
    final data = Map<String, dynamic>.from(_parse(response, 'Get hostel rules') as Map);
    return Map<String, dynamic>.from(data['settings'] as Map? ?? const {});
  }

  Future<Map<String, dynamic>> saveOwnerHostelRules(
    int hostelId, {
    required List<String> rules,
    Map<String, dynamic> messTimings = const {},
    Map<String, dynamic> gatePolicy = const {},
    Map<String, dynamic> announcementPolicy = const {},
  }) async {
    final response = await http.put(
      Uri.parse('$_base/api/owner/hostels/$hostelId/rules'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'rules': rules,
        'mess_timings': messTimings,
        'gate_policy': gatePolicy,
        'announcement_policy': announcementPolicy,
      }),
    );
    final data = Map<String, dynamic>.from(_parse(response, 'Save hostel rules') as Map);
    return Map<String, dynamic>.from(data['settings'] as Map? ?? const {});
  }

  Future<String> getOwnerExportCsv(String kind) async {
    final response = await http.get(
      Uri.parse('$_base/api/owner/exports/$kind'),
      headers: await _authHeaders(),
    );
    if (response.statusCode < 200 || response.statusCode >= 300) {
      _parse(response, 'Export $kind');
    }
    return response.body;
  }

  Future<List<dynamic>> getOwnerBookings() async {
    final response = await http.get(
      Uri.parse('$_base/api/owner/bookings'),
      headers: await _authHeaders(),
    );

    return _parse(response, 'GetOwnerBookings') as List<dynamic>;
  }

  Future<Map<String, dynamic>> updateOwnerBooking(
    int bookingId,
    String status,
  ) async {
    final response = await http.patch(
      Uri.parse('$_base/api/owner/bookings/$bookingId'),
      headers: await _authHeaders(),
      body: jsonEncode({'status': status}),
    );

    return Map<String, dynamic>.from(
      _parse(response, 'UpdateOwnerBooking') as Map,
    );
  }

  Future<Map<String, dynamic>> issueOwnerHostelCredential(int bookingId) async {
    final response = await http.post(
      Uri.parse(
        '$_base/api/owner/bookings/$bookingId/hostel-access-credentials',
      ),
      headers: await _authHeaders(),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'IssueOwnerHostelCredential') as Map,
    );
  }

  Future<Map<String, dynamic>> issueOwnerParentCredential(
    int bookingId, {
    required String parentName,
    required String parentEmail,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/owner/bookings/$bookingId/parent-credentials'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'parent_name': parentName.trim(),
        'parent_email': parentEmail.trim().toLowerCase(),
      }),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'IssueOwnerParentCredential') as Map,
    );
  }

  Future<List<dynamic>> getOwnerParentCredentials() async {
    final response = await http.get(
      Uri.parse('$_base/api/owner/parent-credentials'),
      headers: await _authHeaders(),
    );
    final data = Map<String, dynamic>.from(
      _parse(response, 'GetOwnerParentCredentials') as Map,
    );
    return List<dynamic>.from(data['parents'] as List? ?? const []);
  }

  Future<List<dynamic>> getOwnerComplaints() async {
    final response = await http.get(
      Uri.parse('$_base/api/owner/complaints'),
      headers: await _authHeaders(),
    );

    return _parse(response, 'GetOwnerComplaints') as List<dynamic>;
  }

  // ══════════════════════════════════════════════════════════════════
  // WARDEN APIs
  // ══════════════════════════════════════════════════════════════════

  // ------------------------------------------------------------------
  // DASHBOARD
  // ------------------------------------------------------------------

  Future<Map<String, dynamic>> getWardenDashboard() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/dashboard'),
      headers: await _authHeaders(),
    );

    return _parse(r, 'GetWardenDashboard') as Map<String, dynamic>;
  }

  // ------------------------------------------------------------------
  // FEES + RENT
  // ------------------------------------------------------------------

  Future<List<dynamic>> getWardenFeeRecords(String billingMonth) async {
    final uri = Uri.parse(
      '$_base/api/warden/fee-records',
    ).replace(queryParameters: {'billing_month': billingMonth});
    final response = await http.get(uri, headers: await _authHeaders());
    final data = Map<String, dynamic>.from(
      _parse(response, 'GetWardenFeeRecords') as Map,
    );
    return List<dynamic>.from(data['records'] as List? ?? const []);
  }

  Future<Map<String, dynamic>> setupWardenRentRecord({
    required int bookingId,
    required String billingMonth,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/warden/fee-records/$bookingId/setup'),
      headers: await _authHeaders(),
      body: jsonEncode({'billing_month': billingMonth}),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'SetupWardenRentRecord') as Map,
    );
  }

  Future<Map<String, dynamic>> getWardenStudentRentRecords(
    int studentId,
  ) async {
    final response = await http.get(
      Uri.parse('$_base/api/warden/students/$studentId/rent-records'),
      headers: await _authHeaders(),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'GetWardenStudentRentRecords') as Map,
    );
  }

  Future<Map<String, dynamic>> recordWardenCashFeePayment({
    required int invoiceId,
    required double amount,
    required String receivedOn,
    String? note,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/warden/cash-fee-payments'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'invoice_id': invoiceId,
        'amount': amount,
        'received_on': receivedOn,
        if (note != null && note.trim().isNotEmpty) 'note': note.trim(),
      }),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'RecordWardenCashFeePayment') as Map,
    );
  }

  Future<Map<String, dynamic>> updateWardenCashFeePayment({
    required int paymentId,
    required double amount,
    required String receivedOn,
    String? note,
  }) async {
    final response = await http.patch(
      Uri.parse('$_base/api/warden/cash-fee-payments/$paymentId'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'amount': amount,
        'received_on': receivedOn,
        if (note != null && note.trim().isNotEmpty) 'note': note.trim(),
      }),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'UpdateWardenCashFeePayment') as Map,
    );
  }

  Future<http.Response> downloadFeeReceipt(int receiptId) async {
    final response = await http.get(
      Uri.parse('$_base/api/fee-receipts/$receiptId/download'),
      headers: await _authHeaders(),
    );
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception('Unable to download the uploaded receipt.');
    }
    return response;
  }

  Future<Map<String, dynamic>> reviewWardenFeeReceipt({
    required int receiptId,
    required String status,
    required String note,
  }) async {
    final response = await http.patch(
      Uri.parse('$_base/api/warden/fee-receipts/$receiptId'),
      headers: await _authHeaders(),
      body: jsonEncode({'status': status, 'review_note': note.trim()}),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'ReviewWardenFeeReceipt') as Map,
    );
  }

  // ------------------------------------------------------------------
  // COMPLAINTS
  // ------------------------------------------------------------------

  Future<List<dynamic>> getWardenComplaints() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/complaints'),
      headers: await _authHeaders(),
    );

    return _parse(r, 'GetWardenComplaints') as List<dynamic>;
  }

  Future<Map<String, dynamic>> analyzeWardenComplaint({
    required String description,
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/warden/complaint-analysis'),
      headers: await _authHeaders(),
      body: jsonEncode({'description': description.trim()}),
    );
    return Map<String, dynamic>.from(
      _parse(r, 'AnalyzeWardenComplaint') as Map,
    );
  }

  Future<Map<String, dynamic>> createWardenComplaint({
    required int studentId,
    required String category,
    required String severity,
    required String description,
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/warden/complaints'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'student_id': studentId,
        'category': category,
        'severity': severity.toLowerCase(),
        'description': description,
      }),
    );
    return Map<String, dynamic>.from(_parse(r, 'CreateWardenComplaint') as Map);
  }

  Future<Map<String, dynamic>> updateWardenComplaint({
    required int complaintId,
    required String status,
    String? note,
    String? safeResolutionSummary,
    String? actionsTaken,
    String? preventionFollowUpNotes,
  }) async {
    final r = await http.patch(
      Uri.parse('$_base/api/warden/complaints/$complaintId'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'status': status,
        if (note != null && note.trim().isNotEmpty) 'note': note.trim(),
        if (safeResolutionSummary != null &&
            safeResolutionSummary.trim().isNotEmpty)
          'safe_resolution_summary': safeResolutionSummary.trim(),
        if (actionsTaken != null && actionsTaken.trim().isNotEmpty)
          'actions_taken': actionsTaken.trim(),
        if (preventionFollowUpNotes != null &&
            preventionFollowUpNotes.trim().isNotEmpty)
          'prevention_follow_up_notes': preventionFollowUpNotes.trim(),
      }),
    );

    return _parse(r, 'UpdateWardenComplaint') as Map<String, dynamic>;
  }

  // ------------------------------------------------------------------
  // ATTENDANCE
  // ------------------------------------------------------------------

  Future<List<dynamic>> getWardenAttendance({String? date}) async {
    final uri = Uri.parse(
      '$_base/api/warden/attendance',
    ).replace(queryParameters: {if (date != null) 'date': date});

    final r = await http.get(uri, headers: await _authHeaders());

    return _parse(r, 'GetWardenAttendance') as List<dynamic>;
  }

  Future<Map<String, dynamic>> saveWardenAttendance({
    required int studentId,
    required String status,
    String? date,
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/warden/attendance'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'student_id': studentId,
        'status': status,
        if (date != null) 'date': date,
      }),
    );

    return _parse(r, 'SaveWardenAttendance') as Map<String, dynamic>;
  }

  Future<List<dynamic>> getDailyWardenAttendance(String date) async {
    final uri = Uri.parse(
      '$_base/api/warden/daily-attendance',
    ).replace(queryParameters: {'date': date});
    final r = await http.get(uri, headers: await _authHeaders());
    return List<dynamic>.from(_parse(r, 'GetDailyWardenAttendance') as List);
  }

  Future<Map<String, dynamic>> saveDailyWardenAttendance({
    required String date,
    required List<Map<String, dynamic>> records,
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/warden/daily-attendance'),
      headers: await _authHeaders(),
      body: jsonEncode({'date': date, 'records': records}),
    );
    return Map<String, dynamic>.from(
      _parse(r, 'SaveDailyWardenAttendance') as Map,
    );
  }

  // ------------------------------------------------------------------
  // QR GATE ATTENDANCE
  // ------------------------------------------------------------------

  Future<Map<String, dynamic>> getWardenAssignedHostel() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/assigned-hostel'),
      headers: await _authHeaders(),
    );
    return Map<String, dynamic>.from(
      _parse(r, 'GetWardenAssignedHostel') as Map,
    );
  }

  Future<Map<String, dynamic>> createGateQr({
    required int hostelId,
    String gateLabel = 'Main Gate',
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/warden/hostels/$hostelId/gate-qr'),
      headers: await _authHeaders(),
      body: jsonEncode({'gate_label': gateLabel.trim()}),
    );
    return Map<String, dynamic>.from(_parse(r, 'CreateGateQr') as Map);
  }

  Future<List<dynamic>> getGateQrs(int hostelId) async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/hostels/$hostelId/gate-qr'),
      headers: await _authHeaders(),
    );
    final data = Map<String, dynamic>.from(_parse(r, 'GetGateQrs') as Map);
    return List<dynamic>.from(data['gates'] ?? const []);
  }

  Future<Map<String, dynamic>> getQrAttendanceDashboard(
    int hostelId, {
    String? date,
  }) async {
    final uri = Uri.parse('$_base/api/warden/attendance/dashboard').replace(
      queryParameters: {
        'hostel_id': '$hostelId',
        if (date != null) 'date': date,
      },
    );
    final r = await http.get(uri, headers: await _authHeaders());
    return Map<String, dynamic>.from(
      _parse(r, 'GetQrAttendanceDashboard') as Map,
    );
  }

  /// Manual and QR gate scans are stored together as attendance events.
  /// A warden uses this only when a student could not complete a QR scan.
  Future<Map<String, dynamic>> recordManualGateAttendance({
    required int hostelId,
    required int studentId,
    required String direction,
    required DateTime occurredAt,
  }) async {
    final idempotencyKey =
        'warden-${hostelId}-${studentId}-${direction}-${DateTime.now().microsecondsSinceEpoch}';
    final r = await http.post(
      Uri.parse('$_base/api/warden/attendance/events'),
      headers: {...await _authHeaders(), 'Idempotency-Key': idempotencyKey},
      body: jsonEncode({
        'hostel_id': hostelId,
        'student_id': studentId,
        'direction': direction,
        'occurred_at': occurredAt.toUtc().toIso8601String(),
        'entry_point': 'Warden manual correction',
      }),
    );
    return Map<String, dynamic>.from(
      _parse(r, 'RecordManualGateAttendance') as Map,
    );
  }

  Future<List<dynamic>> getStudentGateAttendance() async {
    final r = await http.get(
      Uri.parse('$_base/api/student/attendance/gate-requests'),
      headers: await _authHeaders(),
    );
    final data = Map<String, dynamic>.from(
      _parse(r, 'GetStudentGateAttendance') as Map,
    );
    return List<dynamic>.from(data['requests'] ?? const []);
  }

  Future<Map<String, dynamic>> getStudentQrAttendanceAccess() async {
    final r = await http.get(
      Uri.parse('$_base/api/student/attendance/qr-access'),
      headers: await _authHeaders(),
    );
    return Map<String, dynamic>.from(
      _parse(r, 'GetStudentQrAttendanceAccess') as Map,
    );
  }

  Future<Map<String, dynamic>> submitGateAttendance({
    required String gateCode,
    required String direction,
    double? latitude,
    double? longitude,
    double? accuracyMeters,
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/student/attendance/gate-requests'),
      headers: {
        ...await _authHeaders(),
        'Idempotency-Key':
            'mobile-${DateTime.now().microsecondsSinceEpoch}-${direction.toLowerCase()}',
      },
      body: jsonEncode({
        'gate_code': gateCode,
        'direction': direction.toLowerCase(),
        if (latitude != null && longitude != null && accuracyMeters != null)
          'location': {
            'latitude': latitude,
            'longitude': longitude,
            'accuracy_m': accuracyMeters,
          },
        if (latitude != null && longitude != null && accuracyMeters != null)
          'location_consent': true,
      }),
    );
    return Map<String, dynamic>.from(_parse(r, 'SubmitGateAttendance') as Map);
  }

  // ------------------------------------------------------------------
  // CONSENTED STUDENT LOCATION SHARING
  // ------------------------------------------------------------------

  Future<List<dynamic>> getStudentLocationSharingSessions() async {
    final response = await http.get(
      Uri.parse('$_base/api/student/location-sharing/sessions'),
      headers: await _authHeaders(),
    );
    final data = Map<String, dynamic>.from(
      _parse(response, 'GetStudentLocationSharingSessions') as Map,
    );
    return List<dynamic>.from(data['sessions'] as List? ?? const []);
  }

  Future<Map<String, dynamic>> startStudentLocationSharing({
    required int durationMinutes,
    bool historyEnabled = false,
    bool geofencingEnabled = false,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/student/location-sharing/sessions'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'duration_minutes': durationMinutes,
        'history_enabled': historyEnabled,
        'geofencing_enabled': geofencingEnabled,
      }),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'StartStudentLocationSharing') as Map,
    );
  }

  Future<void> sendStudentLocationUpdate({
    required int sessionId,
    required double latitude,
    required double longitude,
    double? accuracyMeters,
  }) async {
    final response = await http.post(
      Uri.parse(
        '$_base/api/student/location-sharing/sessions/$sessionId/updates',
      ),
      headers: {
        ...await _authHeaders(),
        'Idempotency-Key': 'location-${DateTime.now().microsecondsSinceEpoch}',
      },
      body: jsonEncode({
        'latitude': latitude,
        'longitude': longitude,
        if (accuracyMeters != null) 'accuracy_m': accuracyMeters,
      }),
    );
    _parse(response, 'SendStudentLocationUpdate');
  }

  Future<void> revokeStudentLocationSharing(int sessionId) async {
    final response = await http.post(
      Uri.parse(
        '$_base/api/student/location-sharing/sessions/$sessionId/revoke',
      ),
      headers: await _authHeaders(),
    );
    _parse(response, 'RevokeStudentLocationSharing');
  }

  Future<List<dynamic>> getParentChildren() async {
    final response = await http.get(
      Uri.parse('$_base/api/parent/children'),
      headers: await _authHeaders(),
    );
    final data = Map<String, dynamic>.from(
      _parse(response, 'GetParentChildren') as Map,
    );
    return List<dynamic>.from(data['children'] as List? ?? const []);
  }

  Future<Map<String, dynamic>> getParentChildOverview(int studentId) async {
    final response = await http.get(
      Uri.parse('$_base/api/parent/children/$studentId/overview'),
      headers: await _authHeaders(),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'GetParentChildOverview') as Map,
    );
  }

  Future<Map<String, dynamic>> getParentChildLocation(int studentId) async {
    final response = await http.get(
      Uri.parse('$_base/api/parent/children/$studentId/location'),
      headers: await _authHeaders(),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'GetParentChildLocation') as Map,
    );
  }

  Future<List<dynamic>> getParentChildTransport(int studentId) async {
    final response = await http.get(
      Uri.parse('$_base/api/parent/children/$studentId/transport'),
      headers: await _authHeaders(),
    );
    final data = Map<String, dynamic>.from(
      _parse(response, 'GetParentChildTransport') as Map,
    );
    return List<dynamic>.from(data['assignments'] as List? ?? const []);
  }

  Future<List<dynamic>> getStudentTransport() async {
    final response = await http.get(
      Uri.parse('$_base/api/student/transport'),
      headers: await _authHeaders(),
    );
    final data = Map<String, dynamic>.from(
      _parse(response, 'GetStudentTransport') as Map,
    );
    return List<dynamic>.from(data['assignments'] as List? ?? const []);
  }

  Future<Map<String, dynamic>> openParentStudentConversation(int studentId) async {
    final response = await http.post(
      Uri.parse('$_base/api/parent/conversations'),
      headers: await _authHeaders(),
      body: jsonEncode({'student_id': studentId, 'type': 'parent_student'}),
    );
    final data = Map<String, dynamic>.from(
      _parse(response, 'OpenParentStudentConversation') as Map,
    );
    return Map<String, dynamic>.from(data['conversation'] as Map);
  }

  Future<List<dynamic>> getConversationMessages(int conversationId) async {
    final response = await http.get(
      Uri.parse('$_base/api/conversations/$conversationId/messages'),
      headers: await _authHeaders(),
    );
    return List<dynamic>.from(
      _parse(response, 'GetConversationMessages') as List? ?? const [],
    );
  }

  Future<Map<String, dynamic>> sendConversationMessage({
    required int conversationId,
    required String body,
  }) async {
    final response = await http.post(
      Uri.parse('$_base/api/conversations/$conversationId/messages'),
      headers: await _authHeaders(),
      body: jsonEncode({'body': body.trim()}),
    );
    final data = Map<String, dynamic>.from(
      _parse(response, 'SendConversationMessage') as Map,
    );
    return Map<String, dynamic>.from(data['message'] as Map);
  }

  Future<Map<String, dynamic>> runParentVoiceCommand(String transcript) async {
    final response = await http.post(
      Uri.parse('$_base/api/parent/voice/command'),
      headers: await _authHeaders(),
      body: jsonEncode({'transcript': transcript.trim()}),
    );
    return Map<String, dynamic>.from(
      _parse(response, 'RunParentVoiceCommand') as Map,
    );
  }

  Future<List<dynamic>> getParentChildEmergencies(int studentId) async {
    final response = await http.get(
      Uri.parse('$_base/api/parent/children/$studentId/emergencies'),
      headers: await _authHeaders(),
    );
    final data = Map<String, dynamic>.from(
      _parse(response, 'GetParentChildEmergencies') as Map,
    );
    return List<dynamic>.from(data['incidents'] as List? ?? const []);
  }

  // ------------------------------------------------------------------
  // ANNOUNCEMENTS
  // ------------------------------------------------------------------

  Future<List<dynamic>> getWardenAnnouncements() async {
    final r = await http.get(
      Uri.parse('$_base/api/announcements'),
      headers: await _authHeaders(),
    );

    return _parse(r, 'GetWardenAnnouncements') as List<dynamic>;
  }

  Future<Map<String, dynamic>> postAnnouncement({
    required int hostelId,
    required String title,
    required String body,
    String audience = 'residents',
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/announcements'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'hostel_id': hostelId,
        'title': title.trim(),
        'body': body.trim(),
        'audience': audience,
      }),
    );

    return _parse(r, 'PostAnnouncement') as Map<String, dynamic>;
  }

  // ------------------------------------------------------------------
  // PENALTIES
  // ------------------------------------------------------------------

  Future<List<dynamic>> getWardenPenalties() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/penalties'),
      headers: await _authHeaders(),
    );

    return _parse(r, 'GetWardenPenalties') as List<dynamic>;
  }

  Future<Map<String, dynamic>> assignPenalty({
    required int studentId,
    required String reason,
    required double amount,
    String? note,
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/warden/penalties'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'student_id': studentId,
        'reason': reason.trim(),
        'amount': amount,
        if (note != null && note.trim().isNotEmpty) 'note': note.trim(),
      }),
    );

    return _parse(r, 'AssignPenalty') as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> markWardenPenaltyPaid(int penaltyId) async {
    final r = await http.patch(
      Uri.parse('$_base/api/warden/penalties/$penaltyId/paid'),
      headers: await _authHeaders(),
    );
    return Map<String, dynamic>.from(_parse(r, 'MarkWardenPenaltyPaid') as Map);
  }

  // ------------------------------------------------------------------
  // ROOM ALLOCATION
  // ------------------------------------------------------------------

  Future<List<dynamic>> getWardenRooms() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/rooms'),
      headers: await _authHeaders(),
    );

    return _parse(r, 'GetWardenRooms') as List<dynamic>;
  }

  Future<List<dynamic>> getWardenRoomAllocations() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/room-allocations'),
      headers: await _authHeaders(),
    );
    return List<dynamic>.from(_parse(r, 'GetWardenRoomAllocations') as List);
  }

  Future<List<dynamic>> getWardenRoomChangeRequests() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/room-change-requests'),
      headers: await _authHeaders(),
    );
    final data = Map<String, dynamic>.from(
      _parse(r, 'GetWardenRoomChangeRequests') as Map,
    );
    return List<dynamic>.from(data['requests'] as List? ?? const []);
  }

  Future<Map<String, dynamic>> decideWardenRoomChangeRequest({
    required int requestId,
    required String status,
  }) async {
    final r = await http.patch(
      Uri.parse('$_base/api/warden/room-change-requests/$requestId'),
      headers: await _authHeaders(),
      body: jsonEncode({'status': status}),
    );
    return Map<String, dynamic>.from(
      _parse(r, 'DecideWardenRoomChangeRequest') as Map,
    );
  }

  Future<Map<String, dynamic>> allocateRoom({
    required int studentId,
    required int roomId,
    int? bedId,
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/warden/room-allocation'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'student_id': studentId,
        'room_id': roomId,
        if (bedId != null) 'bed_id': bedId,
      }),
    );

    return _parse(r, 'AllocateRoom') as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> changeBookingAllocation({
    required int bookingId,
    required int roomId,
    required String bedNumber,
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/bookings/$bookingId/assign-room'),
      headers: await _authHeaders(),
      body: jsonEncode({'room_id': roomId, 'bed_number': bedNumber}),
    );
    return Map<String, dynamic>.from(
      _parse(r, 'ChangeBookingAllocation') as Map,
    );
  }

  Future<Map<String, dynamic>> deallocateBookingRoom(int bookingId) async {
    final r = await http.delete(
      Uri.parse('$_base/api/bookings/$bookingId/room-allocation'),
      headers: await _authHeaders(),
    );
    return Map<String, dynamic>.from(_parse(r, 'DeallocateBookingRoom') as Map);
  }

  Future<List<dynamic>> getAvailableRoomsForBooking(int bookingId) async {
    final r = await http.get(
      Uri.parse('$_base/api/bookings/$bookingId/available-rooms'),
      headers: await _authHeaders(),
    );
    return List<dynamic>.from(_parse(r, 'GetAvailableRoomsForBooking') as List);
  }

  // ------------------------------------------------------------------
  // STUDENTS + PARENTS
  // ------------------------------------------------------------------

  Future<List<dynamic>> getWardenStudents() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/students'),
      headers: await _authHeaders(),
    );

    return _parse(r, 'GetWardenStudents') as List<dynamic>;
  }

  Future<Map<String, dynamic>> getWardenStudentDetails(int studentId) async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/students/$studentId'),
      headers: await _authHeaders(),
    );

    return _parse(r, 'GetWardenStudentDetails') as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> createWardenStudent(
    Map<String, dynamic> student,
  ) async {
    final r = await http.post(
      Uri.parse('$_base/api/warden/students'),
      headers: await _authHeaders(),
      body: jsonEncode(student),
    );
    return Map<String, dynamic>.from(_parse(r, 'CreateWardenStudent') as Map);
  }

  Future<void> updateWardenStudent(
    int studentId,
    Map<String, dynamic> student,
  ) async {
    final r = await http.put(
      Uri.parse('$_base/api/warden/students/$studentId'),
      headers: await _authHeaders(),
      body: jsonEncode(student),
    );
    _parse(r, 'UpdateWardenStudent');
  }

  // ------------------------------------------------------------------
  // EMPTY BEDS
  // ------------------------------------------------------------------

  Future<List<dynamic>> getWardenEmptyBeds() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/empty-beds'),
      headers: await _authHeaders(),
    );

    return _parse(r, 'GetWardenEmptyBeds') as List<dynamic>;
  }

  Future<Map<String, dynamic>> notifyEmptyBed({
    required int roomId,
    int? bedId,
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/warden/empty-beds/notify'),
      headers: await _authHeaders(),
      body: jsonEncode({'room_id': roomId, if (bedId != null) 'bed_id': bedId}),
    );

    return _parse(r, 'NotifyEmptyBed') as Map<String, dynamic>;
  }

  // ------------------------------------------------------------------
  // CHAT
  // ------------------------------------------------------------------

  Future<List<dynamic>> getWardenConversations() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/conversations'),
      headers: await _authHeaders(),
    );

    return _parse(r, 'GetWardenConversations') as List<dynamic>;
  }

  Future<Map<String, dynamic>> openWardenConversation(int studentId) async {
    final r = await http.post(
      Uri.parse('$_base/api/warden/conversations'),
      headers: await _authHeaders(),
      body: jsonEncode({'student_id': studentId}),
    );

    return Map<String, dynamic>.from(
      _parse(r, 'OpenWardenConversation') as Map,
    );
  }

  Future<List<dynamic>> getWardenMessages(int conversationId) async {
    final r = await http.get(
      Uri.parse(
        '$_base/api/warden/conversations/'
        '$conversationId/messages',
      ),
      headers: await _authHeaders(),
    );

    return _parse(r, 'GetWardenMessages') as List<dynamic>;
  }

  Future<Map<String, dynamic>> sendWardenMessage({
    required int conversationId,
    required String message,
  }) async {
    final r = await http.post(
      Uri.parse(
        '$_base/api/warden/conversations/'
        '$conversationId/messages',
      ),
      headers: await _authHeaders(),
      body: jsonEncode({'message': message.trim()}),
    );

    return _parse(r, 'SendWardenMessage') as Map<String, dynamic>;
  }

  // ------------------------------------------------------------------
  // CLEANING SCHEDULE
  // ------------------------------------------------------------------

  Future<List<dynamic>> getCleaningSchedule() async {
    final r = await http.get(
      Uri.parse('$_base/api/warden/cleaning-schedule'),
      headers: await _authHeaders(),
    );

    return _parse(r, 'GetCleaningSchedule') as List<dynamic>;
  }

  Future<Map<String, dynamic>> createCleaningSchedule({
    required String area,
    required String scheduledDate,
    required String shift,
  }) async {
    final r = await http.post(
      Uri.parse('$_base/api/warden/cleaning-schedule'),
      headers: await _authHeaders(),
      body: jsonEncode({
        'area': area.trim(),
        'scheduled_date': scheduledDate,
        'shift': shift,
      }),
    );

    return _parse(r, 'CreateCleaningSchedule') as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> updateCleaningSchedule({
    required int scheduleId,
    required String status,
  }) async {
    final r = await http.patch(
      Uri.parse(
        '$_base/api/warden/cleaning-schedule/'
        '$scheduleId',
      ),
      headers: await _authHeaders(),
      body: jsonEncode({'status': status}),
    );

    return _parse(r, 'UpdateCleaningSchedule') as Map<String, dynamic>;
  }
}
