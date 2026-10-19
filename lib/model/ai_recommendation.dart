class AiRecommendation {
  final String hostelId;
  final String hostelName;
  final String hostelType;
  final String area;
  final double latitude;
  final double longitude;
  final double singleRoomPrice;
  final double distanceFromFastKm;
  final double overallRating;
  final double hybridScore;
  final double cbScore;
  final double cfScore;
  final String studentType;
  final double alphaUsed;
  final List<String> reasons;

  AiRecommendation({
    required this.hostelId,
    required this.hostelName,
    required this.hostelType,
    required this.area,
    required this.latitude,
    required this.longitude,
    required this.singleRoomPrice,
    required this.distanceFromFastKm,
    required this.overallRating,
    required this.hybridScore,
    required this.cbScore,
    required this.cfScore,
    required this.studentType,
    required this.alphaUsed,
    required this.reasons,
  });

  factory AiRecommendation.fromJson(Map<String, dynamic> json) {
    double number(dynamic value) => value is num ? value.toDouble() : double.tryParse(value?.toString() ?? '') ?? 0;
    return AiRecommendation(
      hostelId: (json['hostel_id'] ?? json['id']).toString(),
      hostelName: (json['hostel_name'] ?? json['name'] ?? 'Hostel').toString(),
      hostelType: (json['hostel_type'] ?? 'Hostel').toString(),
      area: (json['area'] ?? json['city'] ?? '').toString(),
      latitude: number(json['latitude']), longitude: number(json['longitude']),
      singleRoomPrice: number(json['single_room_price'] ?? json['double_room_price'] ?? json['dorm_room_price']),
      distanceFromFastKm: number(json['distance_from_fast_km']),
      overallRating: number(json['overall_rating'] ?? json['average_rating']),
      hybridScore: number(json['hybrid_score'] ?? json['model_score']),
      cbScore: number(json['cb_score'] ?? json['content_score']),
      cfScore: number(json['cf_score'] ?? json['collaborative_score']),
      studentType: (json['student_type'] ?? json['model'] ?? 'live profile').toString(),
      alphaUsed: number(json['alpha_used']),
      reasons: List<String>.from(json['reasons'] as List? ?? const []),
    );
  }
}
