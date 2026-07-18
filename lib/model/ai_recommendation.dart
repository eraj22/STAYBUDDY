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
  });

  factory AiRecommendation.fromJson(Map<String, dynamic> json) {
    double number(String primaryKey, [String? fallbackKey]) {
      final value =
          json[primaryKey] ??
          (fallbackKey == null ? null : json[fallbackKey]) ??
          0;
      return (value as num).toDouble();
    }

    return AiRecommendation(
      hostelId: (json['id'] ?? json['hostel_id']).toString(),
      hostelName: (json['name'] ?? json['hostel_name'] ?? '').toString(),
      hostelType: (json['hostel_type'] ?? '').toString(),
      area: (json['area'] ?? '').toString(),
      latitude: number('latitude'),
      longitude: number('longitude'),
      singleRoomPrice: number('single_room_price'),
      distanceFromFastKm: number('distance_from_fast_km'),
      overallRating: number('overall_rating', 'average_rating'),
      hybridScore: number('model_score', 'hybrid_score'),
      cbScore: number('content_score', 'cb_score'),
      cfScore: number('collaborative_score', 'cf_score'),
      studentType: (json['student_type'] ?? '').toString(),
      alphaUsed: number('alpha_used'),
    );
  }
}
