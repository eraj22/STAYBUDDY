import 'package:flutter/material.dart';

import 'owner/wardens/owner_wardens_screen.dart';

/// Legacy compatibility screen.
///
/// Older parts of the StayBuddy application may still navigate to
/// OwnerManageWardensScreen.
///
/// The actual Warden management UI now lives in:
///
/// lib/screens/owner/wardens/owner_wardens_screen.dart
///
/// Keeping this wrapper prevents older navigation code from breaking.
class OwnerManageWardensScreen extends StatelessWidget {
  final String ownerName;
  final String hostelName;
  final int? hostelId;

  const OwnerManageWardensScreen({
    super.key,
    required this.ownerName,
    required this.hostelName,
    this.hostelId,
  });

  @override
  Widget build(BuildContext context) {
    return OwnerWardensScreen(
      hostelName: hostelName,
    );
  }
}