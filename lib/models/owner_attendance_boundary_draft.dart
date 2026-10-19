/// Holds the boundary only while a new owner completes the registration flow.
/// It is written to the backend immediately after the hostel record is created.
class OwnerAttendanceBoundaryDraft {
  static List<Map<String, double>> _points = const [];

  static List<Map<String, double>> get points => List.unmodifiable(_points);

  static void replace(List<Map<String, double>> points) {
    _points = points
        .map(
          (point) => {
            'latitude': point['latitude']!,
            'longitude': point['longitude']!,
          },
        )
        .toList(growable: false);
  }

  static void clear() => _points = const [];
}
