import '../model/ai_recommendation.dart';
import '../api.dart';

class AiRecommendationService {
  static Future<List<AiRecommendation>> fetchRecommendations({
    required String gender,
    required String department,
    required int budgetMax,
    required double maxDistanceKm,
    required double studyPreference,
    required String roomType,
    required double priceSensitivity,
    required double comfortPreference,
    required double noiseTolerance,
    required double curfewFlexibility,
    required bool needsTransport,
    required List<String> mustHave,
    int topK = 5,
  }) async {
    final data = await Api().getLiveMlRecommendations(
      topK: topK,
      budgetMax: budgetMax,
      requiredAmenities: mustHave,
    );
    final list = data["recommendations"] as List<dynamic>;

    return list
        .map((item) => AiRecommendation.fromJson(item as Map<String, dynamic>))
        .toList();
  }
}
