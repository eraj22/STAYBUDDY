import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../api.dart';
import '../routes.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'student_hostel_hub_screen.dart';

/// Internal student landing page. It is opened only when the backend reports
/// a confirmed booking with an assigned room. All displayed values come from
/// the student's access endpoint; there is no fallback/demo assignment.
class StudentHostelAccessScreen extends StatefulWidget {
  const StudentHostelAccessScreen({
    super.key,
    this.openHubImmediately = false,
  });

  final bool openHubImmediately;

  @override
  State<StudentHostelAccessScreen> createState() =>
      _StudentHostelAccessScreenState();
}

class _StudentHostelAccessScreenState extends State<StudentHostelAccessScreen> {
  bool _loading = true;
  String? _error;
  List<Map<String, dynamic>> _assignments = [];
  bool _openedInitialHub = false;

  @override
  void initState() {
    super.initState();
    _loadAccess();
  }

  Future<void> _loadAccess() async {
    try {
      final result = await Api().getStudentAccess();
      final raw = result['assignments'];
      if (!mounted) return;
      setState(() {
        _assignments = raw is List
            ? raw
                  .whereType<Map>()
                  .map((e) => Map<String, dynamic>.from(e))
                  .toList()
            : [];
        _loading = false;
      });
      if (widget.openHubImmediately && !_openedInitialHub && _assignments.isNotEmpty) {
        _openedInitialHub = true;
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (!mounted) return;
          Navigator.of(context).pushReplacement(
            MaterialPageRoute(
              builder: (_) => StudentHostelHubScreen(assignment: _assignments.first),
            ),
          );
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = e.toString().replaceFirst('Exception: ', '');
      });
    }
  }

  Future<void> _logout() async {
    await Api().logout();
    if (!mounted) return;
    Navigator.of(context).pushNamedAndRemoveUntil(Routes.login, (_) => false);
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: .48,
          tintColor: AppTheme.tealDeep,
          tintOpacity: .30,
          cardMargin: const EdgeInsets.all(12),
          cardRadius: 30,
          cardColor: AppTheme.bg.withOpacity(.97),
          child: SafeArea(
            child: _loading
                ? const Center(
                    child: CircularProgressIndicator(color: AppTheme.teal),
                  )
                : _error != null
                ? _errorState()
                : _content(),
          ),
        ),
      ),
    );
  }

  Widget _content() {
    return RefreshIndicator(
      color: AppTheme.teal,
      onRefresh: _loadAccess,
      child: ListView(
        padding: const EdgeInsets.fromLTRB(20, 14, 20, 28),
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  'My Hostel',
                  style: GoogleFonts.playfairDisplay(
                    fontSize: 28,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary,
                  ),
                ),
              ),
              IconButton(
                onPressed: _logout,
                icon: const Icon(Icons.logout_rounded, color: AppTheme.teal),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            'Internal access is unlocked for your confirmed room assignment.',
            style: GoogleFonts.dmSans(
              fontSize: 13,
              color: AppTheme.textSecondary,
            ),
          ),
          const SizedBox(height: 18),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(.10),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppTheme.teal.withOpacity(.25)),
            ),
            child: Row(
              children: [
                const Icon(Icons.verified_user_rounded, color: AppTheme.teal),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'Owner/warden verified membership',
                    style: GoogleFonts.dmSans(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      color: AppTheme.tealDeep,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 18),
          ..._assignments.map(_assignmentCard),
          const SizedBox(height: 18),
          OutlinedButton.icon(
            onPressed: () => Navigator.of(context).pushNamed(Routes.hostels),
            icon: const Icon(Icons.explore_rounded),
            label: const Text('Explore more hostels'),
          ),
        ],
      ),
    );
  }

  Widget _assignmentCard(Map<String, dynamic> item) {
    final hostel = item['hostel_name']?.toString() ?? 'Hostel';
    final floor = item['floor_label']?.toString() ?? '—';
    final room = item['room_number']?.toString() ?? '—';
    final bed = item['bed_number']?.toString() ?? '—';
    final rent = item['monthly_rent'];
    final roomType = item['room_type']?.toString() ?? 'Room';
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(17),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppTheme.border),
        boxShadow: const [
          BoxShadow(
            color: Color(0x12063E40),
            blurRadius: 14,
            offset: Offset(0, 6),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(.10),
                  borderRadius: BorderRadius.circular(13),
                ),
                child: const Icon(
                  Icons.apartment_rounded,
                  color: AppTheme.teal,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  hostel,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    fontSize: 17,
                    fontWeight: FontWeight.w800,
                    color: AppTheme.textPrimary,
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
                decoration: BoxDecoration(
                  color: const Color(0xFFEAF3DE),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  'ACTIVE',
                  style: GoogleFonts.dmSans(
                    fontSize: 10,
                    fontWeight: FontWeight.w800,
                    color: AppTheme.green,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 15),
          Row(
            children: [
              _detail(Icons.layers_rounded, 'Floor', floor),
              _detail(Icons.door_front_door_rounded, 'Room', room),
              _detail(Icons.bed_rounded, 'Bed', bed),
            ],
          ),
          const Divider(height: 24),
          Row(
            children: [
              Icon(Icons.groups_rounded, size: 17, color: AppTheme.teal),
              const SizedBox(width: 7),
              Text(
                roomType,
                style: GoogleFonts.dmSans(
                  fontSize: 12,
                  color: AppTheme.textSecondary,
                ),
              ),
              const Spacer(),
              if (rent != null)
                Text(
                  'PKR $rent / month',
                  style: GoogleFonts.dmSans(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.tealDeep,
                  ),
                ),
            ],
          ),
          const SizedBox(height: 14),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: () => Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (_) => StudentHostelHubScreen(assignment: item),
                ),
              ),
              icon: const Icon(Icons.grid_view_rounded),
              label: const Text('Open Hostel Hub'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _detail(IconData icon, String label, String value) {
    return Expanded(
      child: Row(
        children: [
          Icon(icon, size: 16, color: AppTheme.teal),
          const SizedBox(width: 5),
          Flexible(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: GoogleFonts.dmSans(
                    fontSize: 10,
                    color: AppTheme.textMuted,
                  ),
                ),
                Text(
                  value,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _errorState() => Center(
    child: Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(
            Icons.lock_outline_rounded,
            size: 46,
            color: AppTheme.teal,
          ),
          const SizedBox(height: 12),
          Text(
            'Hostel access could not be loaded',
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 8),
          Text(
            _error!,
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textMuted),
          ),
          const SizedBox(height: 14),
          FilledButton(onPressed: _loadAccess, child: const Text('Try again')),
        ],
      ),
    ),
  );
}
