import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config.dart';
import '../model/ai_recommendation.dart';

class AiRecommendationService {
  static Future<List<AiRecommendation>> fetchRecommendations({
    required String gender,
    required String department,
    required int budgetMax,
    required double maxDistanceKm,
    required double studyPreference,
    required String foodPreference,
    required String roomType,
    required double priceSensitivity,
    required double comfortPreference,
    required double noiseTolerance,
    required double curfewFlexibility,
    required bool needsTransport,
    required List<String> mustHave,
    int topK = 5,
  }) async {
    final uri = Uri.parse('${AppConfig.baseUrl}/api/recommendations/ml');

    final body = {
      "gender": gender,
      "department": department,
      "budget_max": budgetMax,
      "max_distance_km": maxDistanceKm,
      "study_preference": studyPreference,
      "food_preference": foodPreference,
      "room_type": roomType,
      "price_sensitivity": priceSensitivity,
      "comfort_preference": comfortPreference,
      "noise_tolerance": noiseTolerance,
      "curfew_flexibility": curfewFlexibility,
      "needs_transport": needsTransport,
      "must_have": mustHave,
      "top_k": topK,
    };

    final response = await http.post(
      uri,
      headers: {"Content-Type": "application/json"},
      body: jsonEncode(body),
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to fetch AI recommendations: ${response.body}');
    }

    final data = jsonDecode(response.body) as Map<String, dynamic>;
    final list = data['recommendations'];
    if (list is! List) {
      throw Exception('Recommendation service returned an invalid response');
    }

    return list
        .map((item) => AiRecommendation.fromJson(item as Map<String, dynamic>))
        .toList();
  }
}
