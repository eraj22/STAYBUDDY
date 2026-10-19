/// Keeps the detailed accommodation plan while the owner completes onboarding.
/// The complete plan is written to the database only when the hostel is saved.
class OwnerRoomSetupDraft {
  static List<Map<String, dynamic>> _rooms = const [];

  static List<Map<String, dynamic>> get rooms =>
      List.unmodifiable(_rooms.map((room) => Map<String, dynamic>.from(room)));

  static void replace(List<Map<String, dynamic>> value) {
    _rooms = value.map((room) => Map<String, dynamic>.from(room)).toList();
  }

  static void clear() {
    _rooms = const [];
  }
}
