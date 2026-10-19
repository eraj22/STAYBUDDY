import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';

import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';

// =============================================================================
// THEME
// =============================================================================

final Color _teal = AppTheme.teal;
final Color _mint = AppTheme.tealMint;

const Color _glass = Color(0xD909292B);
const Color _orange = Color(0xFFF2A33A);
const Color _red = Color(0xFFE25A5A);
const Color _green = Color(0xFF54C49A);
const Color _purple = Color(0xFF9862D9);

String _allocationDetail({dynamic floor, dynamic room, dynamic bed}) {
  String labelledValue(String label, dynamic value, {bool isFloor = false}) {
    final text = value?.toString().trim();
    if (text == null || text.isEmpty || text == 'null') return '$label —';

    final lowercase = text.toLowerCase();
    if (lowercase.startsWith('${label.toLowerCase()} ') ||
        (isFloor && lowercase.contains('floor'))) {
      return text;
    }
    return '$label $text';
  }

  return '${labelledValue('Floor', floor, isFloor: true)} · '
      '${labelledValue('Room', room)} · '
      '${labelledValue('Bed', bed)}';
}

// =============================================================================
// SHARED WARDEN COMPLAINT STORE
// =============================================================================

class _WardenComplaintStore {
  static final List<Map<String, dynamic>> complaints = [
    {
      'id': 1,
      'student': 'Hira Khan',
      'room': '301',
      'title': 'Fan not working',
      'category': 'Maintenance',
      'subcategory': 'Electrical',
      'priority': 'High',
      'status': 'Pending',
      'suggestion':
          'Assign a technician to inspect the fan and repair or replace faulty electrical parts.',
    },
    {
      'id': 2,
      'student': 'Areeba Fatima',
      'room': '105',
      'title': 'WiFi not connecting',
      'category': 'Internet',
      'subcategory': 'Connectivity',
      'priority': 'High',
      'status': 'Pending',
      'suggestion':
          'Restart the router and inspect network connectivity near Room 105.',
    },
    {
      'id': 3,
      'student': 'Laiba Noor',
      'room': '203',
      'title': 'Washroom tap leaking',
      'category': 'Maintenance',
      'subcategory': 'Plumbing',
      'priority': 'Medium',
      'status': 'Pending',
      'suggestion':
          'Inspect the tap and replace the damaged washer or fitting.',
    },
    {
      'id': 4,
      'student': 'Sara Ali',
      'room': '110',
      'title': 'Broken cupboard lock',
      'category': 'Maintenance',
      'subcategory': 'Furniture',
      'priority': 'Low',
      'status': 'Resolved',
      'suggestion': 'Replace or repair the cupboard lock.',
    },
  ];

  static void addComplaint(Map<String, dynamic> complaint) {
    complaints.insert(0, complaint);
  }
}

// =============================================================================
// COMMON WARDEN PAGE
// =============================================================================

class _WardenPage extends StatelessWidget {
  final String title;
  final String hostelName;
  final Widget child;
  final Widget? floatingActionButton;

  const _WardenPage({
    required this.title,
    required this.hostelName,
    required this.child,
    this.floatingActionButton,
  });

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        resizeToAvoidBottomInset: true,
        floatingActionButton: floatingActionButton,
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.58,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: Column(
              children: [
                _TopBar(title: title, hostelName: hostelName),
                Expanded(child: child),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// TOP BAR
// =============================================================================

class _TopBar extends StatelessWidget {
  final String title;
  final String hostelName;

  const _TopBar({required this.title, required this.hostelName});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 14),
      child: Row(
        children: [
          InkWell(
            onTap: () => Navigator.maybePop(context),
            borderRadius: BorderRadius.circular(50),
            child: Container(
              width: 42,
              height: 42,
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.28),
                shape: BoxShape.circle,
                border: Border.all(color: Colors.white.withOpacity(0.16)),
              ),
              child: const Icon(
                Icons.arrow_back_rounded,
                color: Colors.white,
                size: 20,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  hostelName,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white.withOpacity(0.58),
                    fontSize: 9.5,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// =============================================================================
// COMMON COMPONENTS
// =============================================================================

class _GlassCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;
  final Color? borderColor;

  const _GlassCard({
    required this.child,
    this.padding = const EdgeInsets.all(15),
    this.borderColor,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: padding,
      decoration: BoxDecoration(
        color: _glass,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: borderColor ?? Colors.white.withOpacity(0.11),
        ),
      ),
      child: child,
    );
  }
}

class _PrimaryButton extends StatelessWidget {
  final String text;
  final IconData icon;
  final VoidCallback? onPressed;
  final bool loading;

  const _PrimaryButton({
    required this.text,
    required this.icon,
    required this.onPressed,
    this.loading = false,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: 50,
      child: ElevatedButton(
        onPressed: loading ? null : onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: _teal,
          disabledBackgroundColor: _teal.withOpacity(0.45),
          foregroundColor: Colors.white,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
        child: loading
            ? const SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  color: Colors.white,
                ),
              )
            : Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(icon, size: 18),
                  const SizedBox(width: 8),
                  Flexible(
                    child: Text(
                      text,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(fontWeight: FontWeight.w700),
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}

// =============================================================================
// MOBILE-INTERNAL TOAST
// =============================================================================

void _mobileToast(BuildContext context, String message, {bool error = false}) {
  final overlay = Overlay.maybeOf(context, rootOverlay: false);

  final scaffold = Scaffold.maybeOf(context);

  if (overlay == null || scaffold == null) {
    return;
  }

  late final OverlayEntry entry;

  entry = OverlayEntry(
    builder: (overlayContext) {
      return Positioned(
        left: 16,
        right: 16,
        bottom: 20,
        child: IgnorePointer(
          child: Material(
            color: Colors.transparent,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
              decoration: BoxDecoration(
                color: error ? const Color(0xFF9D3131) : _teal,
                borderRadius: BorderRadius.circular(15),
                border: Border.all(color: Colors.white.withOpacity(0.10)),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.25),
                    blurRadius: 16,
                    offset: const Offset(0, 6),
                  ),
                ],
              ),
              child: Row(
                children: [
                  Icon(
                    error
                        ? Icons.error_outline_rounded
                        : Icons.check_circle_outline_rounded,
                    color: Colors.white,
                    size: 18,
                  ),
                  const SizedBox(width: 9),
                  Expanded(
                    child: Text(
                      message,
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontSize: 10.5,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    },
  );

  overlay.insert(entry);

  Future.delayed(const Duration(seconds: 2), () {
    if (entry.mounted) {
      entry.remove();
    }
  });
}

// =============================================================================
// INPUT DECORATION
// =============================================================================

InputDecoration _inputDecoration(String hint, {IconData? icon}) {
  return InputDecoration(
    hintText: hint,
    hintStyle: GoogleFonts.dmSans(
      color: Colors.white.withOpacity(0.42),
      fontSize: 10,
    ),
    prefixIcon: icon == null
        ? null
        : Icon(icon, size: 18, color: Colors.white.withOpacity(0.55)),
    filled: true,
    fillColor: Colors.black.withOpacity(0.30),
    contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 15),
    border: OutlineInputBorder(
      borderRadius: BorderRadius.circular(15),
      borderSide: BorderSide(color: Colors.white.withOpacity(0.10)),
    ),
    enabledBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(15),
      borderSide: BorderSide(color: Colors.white.withOpacity(0.10)),
    ),
    focusedBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(15),
      borderSide: BorderSide(color: _teal, width: 1.2),
    ),
  );
}

// =============================================================================
// 1. NORMAL COMPLAINTS
// =============================================================================

class WardenComplaintsScreen extends StatefulWidget {
  final String hostelName;

  const WardenComplaintsScreen({super.key, required this.hostelName});

  @override
  State<WardenComplaintsScreen> createState() => _WardenComplaintsScreenState();
}

class _WardenComplaintsScreenState extends State<WardenComplaintsScreen> {
  String _tab = 'All';
  bool _loading = true;
  String? _error;
  List<Map<String, dynamic>> _complaints = [];

  @override
  void initState() {
    super.initState();
    _loadComplaints();
  }

  String _displayStatus(dynamic value) {
    switch ((value ?? '').toString().toLowerCase()) {
      case 'open':
        return 'Pending';
      case 'in_progress':
        return 'In Progress';
      case 'resolved':
        return 'Resolved';
      case 'closed':
        return 'Closed';
      default:
        return (value ?? 'Pending').toString();
    }
  }

  String _displayPriority(Map<String, dynamic> item) {
    final raw = (item['priority'] ?? item['severity'] ?? 'medium')
        .toString()
        .toLowerCase();
    if (raw == 'urgent') return 'Urgent';
    if (raw == 'high') return 'High';
    if (raw == 'low') return 'Low';
    return 'Medium';
  }

  Map<String, dynamic> _normalize(Map<String, dynamic> raw) {
    return {
      ...raw,
      'student': raw['student_name'] ?? 'Student',
      'room': raw['room_number'] ?? raw['room'] ?? '-',
      'floor': raw['floor_label'] ?? raw['floor'] ?? raw['floor_number'],
      'bed': raw['bed_number'] ?? raw['bed'],
      'title': raw['description'] ?? raw['title'] ?? 'Complaint',
      'category': raw['category'] ?? 'Other',
      'subcategory': raw['subcategory'] ?? 'General',
      'priority': _displayPriority(raw),
      'status': _displayStatus(raw['status']),
      'suggestion':
          raw['suggestion'] ??
          'Review this complaint and take the appropriate hostel action.',
    };
  }

  Future<void> _loadComplaints() async {
    if (mounted) {
      setState(() {
        _loading = true;
        _error = null;
      });
    }

    try {
      final rows = await Api().getWardenComplaints();
      final mapped = rows
          .whereType<Map>()
          .map((e) => _normalize(Map<String, dynamic>.from(e)))
          .toList();

      if (!mounted) return;
      setState(() {
        _complaints = mapped;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = e.toString().replaceFirst('Exception: ', '');
      });
    }
  }

  List<Map<String, dynamic>> get _visibleComplaints {
    if (_tab == 'Pending') {
      return _complaints.where((item) => item['status'] == 'Pending').toList();
    }
    if (_tab == 'Resolved') {
      return _complaints.where((item) => item['status'] == 'Resolved').toList();
    }
    return List<Map<String, dynamic>>.from(_complaints);
  }

  Future<void> _accept(Map<String, dynamic> complaint) async {
    await _changeStatus(
      complaint,
      status: 'in_progress',
      successMessage: 'Complaint accepted and marked in progress.',
    );
  }

  Future<void> _resolve(Map<String, dynamic> complaint) async {
    await _changeStatus(
      complaint,
      status: 'resolved',
      successMessage: 'Complaint resolved successfully.',
      safeResolutionSummary:
          'Complaint reviewed and resolved by the assigned warden.',
      actionsTaken:
          'The warden reviewed the complaint and completed the required action.',
    );
  }

  Future<void> _close(Map<String, dynamic> complaint) async {
    await _changeStatus(
      complaint,
      status: 'closed',
      successMessage: 'Complaint closed.',
      safeResolutionSummary:
          'Complaint reviewed and closed by the assigned warden.',
      actionsTaken: 'The warden reviewed the complaint and closed the case.',
    );
  }

  Future<void> _changeStatus(
    Map<String, dynamic> complaint, {
    required String status,
    required String successMessage,
    String? safeResolutionSummary,
    String? actionsTaken,
  }) async {
    final id = int.tryParse('${complaint['id']}');
    if (id == null) {
      _mobileToast(context, 'Invalid complaint ID.', error: true);
      return;
    }

    try {
      await Api().updateWardenComplaint(
        complaintId: id,
        status: status,
        safeResolutionSummary: safeResolutionSummary,
        actionsTaken: actionsTaken,
      );
      await _loadComplaints();
      if (!mounted) return;
      _mobileToast(context, successMessage);
    } catch (e) {
      if (!mounted) return;
      _mobileToast(
        context,
        e.toString().replaceFirst('Exception: ', ''),
        error: true,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: 'Complaints',
      hostelName: widget.hostelName,
      child: RefreshIndicator(
        onRefresh: _loadComplaints,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(
            parent: BouncingScrollPhysics(),
          ),
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
          child: Column(
            children: [
              _complaintStats(),
              const SizedBox(height: 14),
              Row(
                children: [
                  Expanded(child: _filterButton('All')),
                  const SizedBox(width: 7),
                  Expanded(child: _filterButton('Pending')),
                  const SizedBox(width: 7),
                  Expanded(child: _filterButton('Resolved')),
                ],
              ),
              const SizedBox(height: 14),
              if (_loading)
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 42),
                  child: CircularProgressIndicator(),
                )
              else if (_error != null)
                _GlassCard(
                  child: Column(
                    children: [
                      const Icon(Icons.cloud_off_rounded, color: _red),
                      const SizedBox(height: 8),
                      Text(
                        _error!,
                        textAlign: TextAlign.center,
                        style: GoogleFonts.dmSans(
                          color: Colors.white.withOpacity(0.72),
                          fontSize: 9,
                        ),
                      ),
                      const SizedBox(height: 10),
                      TextButton(
                        onPressed: _loadComplaints,
                        child: const Text('Retry'),
                      ),
                    ],
                  ),
                )
              else if (_visibleComplaints.isEmpty)
                _emptyMessage(
                  _tab == 'All' ? 'No complaints' : 'No $_tab complaints',
                )
              else
                ..._visibleComplaints.map(_complaintCard),
            ],
          ),
        ),
      ),
    );
  }

  Widget _complaintStats() {
    final pending = _complaints.where((e) => e['status'] == 'Pending').length;
    final resolved = _complaints.where((e) => e['status'] == 'Resolved').length;
    return _GlassCard(
      child: Row(
        children: [
          _stat('${_complaints.length}', 'Total', Colors.white),
          _stat('$pending', 'Pending', _orange),
          _stat('$resolved', 'Resolved', _mint),
        ],
      ),
    );
  }

  Widget _stat(String value, String label, Color color) {
    return Expanded(
      child: Column(
        children: [
          Text(
            value,
            style: GoogleFonts.dmSans(
              color: color,
              fontSize: 17,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: GoogleFonts.dmSans(
              color: Colors.white.withOpacity(0.5),
              fontSize: 8,
            ),
          ),
        ],
      ),
    );
  }

  Widget _filterButton(String title) {
    final selected = _tab == title;
    return InkWell(
      onTap: () => setState(() => _tab = title),
      borderRadius: BorderRadius.circular(12),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        height: 38,
        decoration: BoxDecoration(
          color: selected
              ? _teal.withOpacity(0.26)
              : Colors.black.withOpacity(0.20),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: selected
                ? _mint.withOpacity(0.42)
                : Colors.white.withOpacity(0.08),
          ),
        ),
        child: Center(
          child: Text(
            title,
            style: GoogleFonts.dmSans(
              color: selected ? _mint : Colors.white.withOpacity(0.66),
              fontSize: 9,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
      ),
    );
  }

  Widget _complaintCard(Map<String, dynamic> complaint) {
    final status = complaint['status']?.toString() ?? 'Pending';
    final resolved = status == 'Resolved';
    final closed = status == 'Closed';
    final inProgress = status == 'In Progress';

    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: _GlassCard(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: _teal.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(13),
                  ),
                  child: Icon(
                    resolved
                        ? Icons.check_circle_outline_rounded
                        : closed
                        ? Icons.cancel_outlined
                        : inProgress
                        ? Icons.build_circle_outlined
                        : Icons.warning_amber_rounded,
                    color: resolved
                        ? _mint
                        : closed
                        ? _red
                        : inProgress
                        ? _teal
                        : _orange,
                  ),
                ),
                const SizedBox(width: 11),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        complaint['title']?.toString() ?? 'Complaint',
                        style: GoogleFonts.dmSans(
                          color: Colors.white,
                          fontWeight: FontWeight.w800,
                          fontSize: 11,
                        ),
                      ),
                      const SizedBox(height: 3),
                      Text(
                        '${complaint['student'] ?? 'Student'} · ${_allocationDetail(floor: complaint['floor'], room: complaint['room'], bed: complaint['bed'])}',
                        style: GoogleFonts.dmSans(
                          color: Colors.white.withOpacity(0.5),
                          fontSize: 8.5,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 5),
                _statusChip(status),
              ],
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 7,
              runSpacing: 7,
              children: [
                _tinyChip('${complaint['category'] ?? 'Other'}'),
                _tinyChip(complaint['subcategory']?.toString() ?? 'General'),
                _tinyChip(complaint['priority']?.toString() ?? 'Medium'),
              ],
            ),
            const SizedBox(height: 12),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: _teal.withOpacity(0.09),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.tips_and_updates_outlined, color: _mint, size: 17),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      complaint['suggestion']?.toString() ??
                          'Review the complaint and take suitable action.',
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.75),
                        fontSize: 9,
                        height: 1.45,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            if (status == 'Pending') ...[
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton(
                      onPressed: () => _close(complaint),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: Colors.white,
                        side: BorderSide(color: Colors.white.withOpacity(0.22)),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(13),
                        ),
                      ),
                      child: const Text('Close'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () => _accept(complaint),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: _teal,
                        foregroundColor: Colors.white,
                        elevation: 0,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(13),
                        ),
                      ),
                      child: const Text('Accept'),
                    ),
                  ),
                ],
              ),
            ] else if (status == 'In Progress') ...[
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () => _resolve(complaint),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _teal,
                    foregroundColor: Colors.white,
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(13),
                    ),
                  ),
                  child: const Text('Resolve'),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _tinyChip(String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
      decoration: BoxDecoration(
        color: _teal.withOpacity(0.13),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Text(
        text,
        style: GoogleFonts.dmSans(
          color: _mint,
          fontSize: 8,
          fontWeight: FontWeight.w700,
        ),
      ),
    );
  }

  Widget _statusChip(String text) {
    Color color = _orange;
    if (text == 'Resolved') {
      color = _green;
    } else if (text == 'Closed') {
      color = _red;
    } else if (text == 'In Progress') {
      color = _teal;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Text(
        text,
        style: GoogleFonts.dmSans(
          color: color,
          fontSize: 8,
          fontWeight: FontWeight.w800,
        ),
      ),
    );
  }
}

// =============================================================================
// AI COMPLAINT ANALYSIS
// =============================================================================

class WardenAiComplaintScreen extends StatefulWidget {
  final String hostelName;

  const WardenAiComplaintScreen({super.key, required this.hostelName});

  @override
  State<WardenAiComplaintScreen> createState() =>
      _WardenAiComplaintScreenState();
}

class _WardenAiComplaintScreenState extends State<WardenAiComplaintScreen> {
  final _studentCtrl = TextEditingController();

  final _roomCtrl = TextEditingController();

  final _complaintCtrl = TextEditingController();
  List<Map<String, dynamic>> _students = [];
  int? _selectedStudentId;
  String? _selectedFloor;
  String? _selectedRoom;
  String? _selectedBed;
  bool _loadingStudents = true;

  bool _analyzing = false;
  bool _analysisReady = false;

  String _category = 'Maintenance';
  String _subcategory = 'Electrical';
  String _priority = 'High';
  String _suggestion =
      'Assign maintenance staff to inspect the issue and repair or replace the faulty electrical component.';

  @override
  void initState() {
    super.initState();
    _loadStudents();
  }

  Future<void> _loadStudents() async {
    try {
      final rows = await Api().getWardenStudents();
      if (!mounted) return;
      setState(() {
        _students = rows
            .whereType<Map>()
            .map((row) => Map<String, dynamic>.from(row))
            .where((student) => student['id'] != null)
            .toList();
        _loadingStudents = false;
      });
    } catch (_) {
      if (mounted) setState(() => _loadingStudents = false);
    }
  }

  @override
  void dispose() {
    _studentCtrl.dispose();
    _roomCtrl.dispose();
    _complaintCtrl.dispose();
    super.dispose();
  }

  Future<void> _analyze() async {
    final text = _complaintCtrl.text.trim();

    if (text.isEmpty || _selectedStudentId == null) {
      _mobileToast(
        context,
        'Select a student and describe the complaint first.',
        error: true,
      );
      return;
    }

    setState(() {
      _analyzing = true;
      _analysisReady = false;
    });

    try {
      final result = await Api().analyzeWardenComplaint(description: text);
      if (!mounted) return;
      final priority = result['priority']?.toString().trim() ?? 'Medium';
      setState(() {
        _category = result['category']?.toString().trim() ?? 'Other';
        _subcategory = result['subcategory']?.toString().trim() ?? 'General';
        _priority = priority.isEmpty ? 'Medium' : priority;
        _suggestion = result['suggestion']?.toString().trim() ?? 'No suggested action was returned by the trained model.';
        _analyzing = false;
        _analysisReady = true;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _analyzing = false);
      _mobileToast(context, error.toString().replaceFirst('Exception: ', ''), error: true);
    }
  }

  Future<void> _addToComplaints() async {
    if (!_analysisReady) {
      return;
    }

    final complaintText = _complaintCtrl.text.trim();

    if (_selectedStudentId == null || complaintText.isEmpty) {
      _mobileToast(
        context,
        'Select a student and enter a complaint first.',
        error: true,
      );
      return;
    }
    try {
      await Api().createWardenComplaint(
        studentId: _selectedStudentId!,
        category: _category,
        severity: _priority,
        description: complaintText,
      );
      final newComplaint = <String, dynamic>{
        'id': DateTime.now().millisecondsSinceEpoch,
        'student': _studentCtrl.text.trim().isEmpty
            ? 'Student'
            : _studentCtrl.text.trim(),
        'room': _roomCtrl.text.trim().isEmpty ? '-' : _roomCtrl.text.trim(),
        'title': complaintText,
        'category': _category,
        'subcategory': _subcategory,
        'priority': _priority,
        'status': 'Pending',
        'suggestion': _suggestion,
      };

      _WardenComplaintStore.addComplaint(newComplaint);

      _studentCtrl.clear();
      _roomCtrl.clear();
      _complaintCtrl.clear();

      setState(() {
        _analysisReady = false;
        _selectedStudentId = null;
        _selectedFloor = null;
        _selectedRoom = null;
        _selectedBed = null;
      });

      _mobileToast(context, 'Complaint saved to complaints list.');
    } catch (error) {
      if (mounted)
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
    }
  }

  void _openComplaintsList() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => WardenComplaintsScreen(hostelName: widget.hostelName),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: 'AI Complaint Analysis',
      hostelName: widget.hostelName,
      child: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
        child: Column(
          children: [
            _GlassCard(
              borderColor: _mint.withOpacity(0.35),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.auto_awesome_rounded, color: _mint, size: 20),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'AI categorizes the complaint, determines priority and recommends a suitable action in one analysis.',
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.75),
                        fontSize: 9.5,
                        height: 1.5,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 14),

            DropdownButtonFormField<int>(
              value: _selectedStudentId,
              isExpanded: true,
              dropdownColor: const Color(0xFF07393C),
              style: GoogleFonts.dmSans(color: Colors.white),
              hint: Text(
                'Select student',
                style: GoogleFonts.dmSans(color: Colors.white.withOpacity(0.8)),
              ),
              iconEnabledColor: Colors.white,
              decoration: _inputDecoration(
                _loadingStudents ? 'Loading students...' : 'Select student',
                icon: Icons.person_outline_rounded,
              ),
              items: _students
                  .map(
                    (student) => DropdownMenuItem<int>(
                      value: int.parse(student['id'].toString()),
                      child: Text(
                        student['name'].toString(),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  )
                  .toList(),
              onChanged: _loadingStudents
                  ? null
                  : (id) {
                      final student = _students.firstWhere(
                        (item) => int.tryParse(item['id'].toString()) == id,
                      );
                      setState(() {
                        _selectedStudentId = id;
                        _studentCtrl.text = student['name'].toString();
                        _roomCtrl.text =
                            student['room_number']?.toString() ?? '';
                        _selectedFloor = student['floor_label']?.toString();
                        _selectedRoom = student['room_number']?.toString();
                        _selectedBed = student['bed_number']?.toString();
                      });
                    },
            ),

            const SizedBox(height: 9),

            DropdownButtonFormField<String>(
              value: _selectedFloor,
              isExpanded: true,
              dropdownColor: const Color(0xFF07393C),
              style: GoogleFonts.dmSans(color: Colors.white),
              hint: Text(
                'Floor number',
                style: GoogleFonts.dmSans(color: Colors.white.withOpacity(0.8)),
              ),
              iconEnabledColor: Colors.white,
              decoration: _inputDecoration(
                'Floor number',
                icon: Icons.layers_outlined,
              ),
              items: _students
                  .map((student) => student['floor_label']?.toString())
                  .whereType<String>()
                  .toSet()
                  .map(
                    (floor) => DropdownMenuItem(
                      value: floor,
                      child: Text('Floor $floor'),
                    ),
                  )
                  .toList(),
              onChanged: (floor) => setState(() {
                _selectedFloor = floor;
                _selectedRoom = null;
                _selectedBed = null;
              }),
            ),

            const SizedBox(height: 9),

            DropdownButtonFormField<String>(
              value: _selectedRoom,
              isExpanded: true,
              dropdownColor: const Color(0xFF07393C),
              style: GoogleFonts.dmSans(color: Colors.white),
              hint: Text(
                'Room number',
                style: GoogleFonts.dmSans(color: Colors.white.withOpacity(0.8)),
              ),
              iconEnabledColor: Colors.white,
              decoration: _inputDecoration(
                'Room number',
                icon: Icons.meeting_room_outlined,
              ),
              items: _students
                  .where(
                    (student) =>
                        _selectedFloor == null ||
                        student['floor_label']?.toString() == _selectedFloor,
                  )
                  .map((student) => student['room_number']?.toString())
                  .whereType<String>()
                  .toSet()
                  .map(
                    (room) => DropdownMenuItem(
                      value: room,
                      child: Text('Room $room'),
                    ),
                  )
                  .toList(),
              onChanged: (room) => setState(() {
                _selectedRoom = room;
                _selectedBed = null;
                _roomCtrl.text = room ?? '';
              }),
            ),

            const SizedBox(height: 9),

            DropdownButtonFormField<String>(
              value: _selectedBed,
              isExpanded: true,
              dropdownColor: const Color(0xFF07393C),
              style: GoogleFonts.dmSans(color: Colors.white),
              hint: Text(
                'Bed number',
                style: GoogleFonts.dmSans(color: Colors.white.withOpacity(0.8)),
              ),
              iconEnabledColor: Colors.white,
              decoration: _inputDecoration(
                'Bed number',
                icon: Icons.bed_outlined,
              ),
              items: _students
                  .where(
                    (student) =>
                        (_selectedFloor == null ||
                            student['floor_label']?.toString() ==
                                _selectedFloor) &&
                        (_selectedRoom == null ||
                            student['room_number']?.toString() ==
                                _selectedRoom),
                  )
                  .map((student) => student['bed_number']?.toString())
                  .whereType<String>()
                  .toSet()
                  .map(
                    (bed) =>
                        DropdownMenuItem(value: bed, child: Text('Bed $bed')),
                  )
                  .toList(),
              onChanged: (bed) {
                final matches = _students
                    .where(
                      (item) =>
                          item['floor_label']?.toString() == _selectedFloor &&
                          item['room_number']?.toString() == _selectedRoom &&
                          item['bed_number']?.toString() == bed,
                    )
                    .toList();
                final student = matches.isEmpty ? null : matches.first;
                setState(() {
                  _selectedBed = bed;
                  if (student != null) {
                    _selectedStudentId = int.tryParse(student['id'].toString());
                    _studentCtrl.text = student['name'].toString();
                  }
                });
              },
            ),

            const SizedBox(height: 10),

            _aiInput(
              controller: _complaintCtrl,
              hint: 'Describe the complaint in detail...',
              icon: Icons.report_problem_outlined,
              maxLines: 5,
            ),

            const SizedBox(height: 14),

            _PrimaryButton(
              text: 'Analyze Complaint',
              icon: Icons.auto_awesome_rounded,
              loading: _analyzing,
              onPressed: _analyze,
            ),

            if (_analysisReady) ...[
              const SizedBox(height: 16),
              _buildAnalysisResult(),
            ],
          ],
        ),
      ),
    );
  }

  Widget _aiInput({
    required TextEditingController controller,
    required String hint,
    required IconData icon,
    int maxLines = 1,
  }) {
    return TextField(
      controller: controller,
      maxLines: maxLines,
      style: GoogleFonts.dmSans(color: Colors.white, fontSize: 11),
      decoration: _inputDecoration(hint, icon: icon),
    );
  }

  Widget _buildAnalysisResult() {
    final priorityColor = _priority == 'High'
        ? _red
        : _priority == 'Medium'
        ? _orange
        : _green;

    return _GlassCard(
      borderColor: _mint.withOpacity(0.45),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.auto_awesome_rounded, color: _mint, size: 19),
              const SizedBox(width: 8),

              Expanded(
                child: Text(
                  'AI Analysis Result',
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),

              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
                decoration: BoxDecoration(
                  color: priorityColor.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  _priority,
                  style: GoogleFonts.dmSans(
                    color: priorityColor,
                    fontSize: 8,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 14),

          Wrap(
            spacing: 7,
            runSpacing: 7,
            children: [
              _resultChip('Category', _category),
              _resultChip('Subcategory', _subcategory),
            ],
          ),

          const SizedBox(height: 14),

          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(13),
            decoration: BoxDecoration(
              color: _teal.withOpacity(0.13),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(
                      Icons.tips_and_updates_outlined,
                      color: _mint,
                      size: 17,
                    ),
                    const SizedBox(width: 7),
                    Text(
                      'Suggested Action',
                      style: GoogleFonts.dmSans(
                        color: _mint,
                        fontWeight: FontWeight.w800,
                        fontSize: 9.5,
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 7),

                Text(
                  _suggestion,
                  style: GoogleFonts.dmSans(
                    color: Colors.white.withOpacity(0.78),
                    fontSize: 9.5,
                    height: 1.5,
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 14),

          _PrimaryButton(
            text: 'Add to Complaints List',
            icon: Icons.add_circle_outline_rounded,
            onPressed: _addToComplaints,
          ),

          const SizedBox(height: 9),

          SizedBox(
            width: double.infinity,
            height: 44,
            child: OutlinedButton.icon(
              onPressed: _openComplaintsList,
              icon: Icon(Icons.list_alt_rounded, color: _mint, size: 17),
              label: Text(
                'View Complaints List',
                style: GoogleFonts.dmSans(
                  color: _mint,
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                ),
              ),
              style: OutlinedButton.styleFrom(
                side: BorderSide(color: _mint.withOpacity(0.35)),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _resultChip(String label, String value) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.07),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: GoogleFonts.dmSans(
              color: Colors.white.withOpacity(0.45),
              fontSize: 7.5,
            ),
          ),
          const SizedBox(height: 3),
          Text(
            value,
            style: GoogleFonts.dmSans(
              color: _mint,
              fontSize: 9.5,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }
}

// =============================================================================
// 2. ATTENDANCE
// =============================================================================

class WardenAttendanceScreen extends StatefulWidget {
  final String hostelName;

  const WardenAttendanceScreen({super.key, required this.hostelName});

  @override
  State<WardenAttendanceScreen> createState() => _WardenAttendanceScreenState();
}

class _WardenAttendanceScreenState extends State<WardenAttendanceScreen> {
  List<Map<String, dynamic>> _students = [];
  final TextEditingController _studentSearchController =
      TextEditingController();
  DateTime _selectedDate = DateTime.now();
  String _selectedFloor = 'All';
  String _selectedRoom = 'All';
  bool _loading = true;
  List<Map<String, dynamic>> _events = [];

  String get _dateKey =>
      '${_selectedDate.year.toString().padLeft(4, '0')}-${_selectedDate.month.toString().padLeft(2, '0')}-${_selectedDate.day.toString().padLeft(2, '0')}';

  @override
  void initState() {
    super.initState();
    _loadAttendance();
  }

  @override
  void dispose() {
    _studentSearchController.dispose();
    super.dispose();
  }

  List<String> get _floorOptions {
    final floors =
        _students
            .map((student) => student['floor']?.toString().trim() ?? '')
            .where((floor) => floor.isNotEmpty && floor != 'null')
            .toSet()
            .toList()
          ..sort();
    return ['All', ...floors];
  }

  List<String> get _roomOptions {
    final rooms =
        _students
            .where(
              (student) =>
                  _selectedFloor == 'All' ||
                  student['floor']?.toString() == _selectedFloor,
            )
            .map((student) => student['room']?.toString().trim() ?? '')
            .where((room) => room.isNotEmpty && room != 'null')
            .toSet()
            .toList()
          ..sort();
    return ['All', ...rooms];
  }

  List<Map<String, dynamic>> get _visibleStudents {
    final search = _studentSearchController.text.trim().toLowerCase();
    return _students.where((student) {
      final name = student['name']?.toString().toLowerCase() ?? '';
      final matchesName = search.isEmpty || name.contains(search);
      final matchesFloor =
          _selectedFloor == 'All' ||
          student['floor']?.toString() == _selectedFloor;
      final matchesRoom =
          _selectedRoom == 'All' ||
          student['room']?.toString() == _selectedRoom;
      return matchesName && matchesFloor && matchesRoom;
    }).toList();
  }

  List<Map<String, dynamic>> get _visibleEvents {
    final search = _studentSearchController.text.trim().toLowerCase();
    return _events.where((event) {
      final name = event['student_name']?.toString().toLowerCase() ?? '';
      final floor = event['floor_label']?.toString() ?? '';
      final room = event['room_number']?.toString() ?? '';
      return (search.isEmpty || name.contains(search)) &&
          (_selectedFloor == 'All' || floor == _selectedFloor) &&
          (_selectedRoom == 'All' || room == _selectedRoom);
    }).toList();
  }

  Future<void> _loadAttendance() async {
    setState(() => _loading = true);
    try {
      final assigned = await Api().getWardenAssignedHostel();
      final hostel = Map<String, dynamic>.from(assigned['hostel'] as Map);
      final hostelId = int.parse(hostel['id'].toString());
      final dashboard = await Api().getQrAttendanceDashboard(
        hostelId,
        date: _dateKey,
      );
      final dailyRows = List<dynamic>.from(
        dashboard['daily_attendance'] as List? ?? const [],
      );
      final dailyByStudent = <String, Map<String, dynamic>>{
        for (final row in dailyRows)
          Map<String, dynamic>.from(row as Map)['student_user_id'].toString():
              Map<String, dynamic>.from(row),
      };
      final residents = List<dynamic>.from(
        dashboard['residents'] as List? ?? const [],
      );
      if (!mounted) return;
      setState(() {
        _students = residents.whereType<Map>().map((row) {
          final item = Map<String, dynamic>.from(row);
          final daily = dailyByStudent[item['student_id'].toString()];
          return {
            'id': item['student_id'],
            'name': item['name'] ?? 'Student',
            'room': item['room_number'],
            'floor': item['floor_label'],
            'bed': item['bed_number'],
            'present': daily != null,
            'direction': daily?['direction'],
            'event_count': daily?['event_count'] ?? 0,
          };
        }).toList();
        _events =
            List<dynamic>.from(dashboard['recent_events'] as List? ?? const [])
                .whereType<Map>()
                .map((row) => Map<String, dynamic>.from(row))
                .toList();
        _selectedFloor = 'All';
        _selectedRoom = 'All';
        _loading = false;
      });
    } catch (error) {
      if (mounted) {
        setState(() => _loading = false);
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
      }
    }
  }

  Future<void> _chooseDate() async {
    DateTime draft = _selectedDate;
    final date = await showDialog<DateTime>(
      context: context,
      useRootNavigator: false,
      builder: (dialogContext) => StatefulBuilder(
        builder: (dialogContext, setDialogState) => Dialog(
          insetPadding: const EdgeInsets.symmetric(
            horizontal: 24,
            vertical: 24,
          ),
          backgroundColor: const Color(0xFFF2F5F5),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(24),
          ),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 340),
            child: Padding(
              padding: const EdgeInsets.fromLTRB(12, 18, 12, 10),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 10),
                    child: Align(
                      alignment: Alignment.centerLeft,
                      child: Text(
                        'Select attendance date',
                        style: GoogleFonts.dmSans(
                          fontWeight: FontWeight.w800,
                          fontSize: 16,
                          color: const Color(0xFF06383A),
                        ),
                      ),
                    ),
                  ),
                  CalendarDatePicker(
                    initialDate: draft,
                    firstDate: DateTime(2024),
                    lastDate: DateTime(2100),
                    onDateChanged: (value) =>
                        setDialogState(() => draft = value),
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      TextButton(
                        onPressed: () => Navigator.pop(dialogContext),
                        child: const Text('Cancel'),
                      ),
                      TextButton(
                        onPressed: () => Navigator.pop(dialogContext, draft),
                        child: const Text('Done'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
    if (date == null || !mounted) return;
    setState(() => _selectedDate = date);
    await _loadAttendance();
  }

  Future<void> _downloadPdf() async {
    final report = pw.Document();
    report.addPage(
      pw.MultiPage(
        build: (_) => [
          pw.Text(
            'StayBuddy Attendance Report',
            style: pw.TextStyle(fontSize: 20, fontWeight: pw.FontWeight.bold),
          ),
          pw.SizedBox(height: 6),
          pw.Text('${widget.hostelName} · Date: $_dateKey'),
          pw.SizedBox(height: 18),
          if (_events.isEmpty)
            pw.Text('No Attendance Record')
          else
            pw.Table.fromTextArray(
              headers: const [
                'Student',
                'Floor',
                'Room',
                'Bed',
                'IN/OUT',
                'Time',
              ],
              data: _events
                  .map(
                    (event) => [
                      event['student_name']?.toString() ?? 'Student',
                      event['floor_label']?.toString() ?? '-',
                      event['room_number']?.toString() ?? '-',
                      event['bed_number']?.toString() ?? '-',
                      event['direction']?.toString().toUpperCase() ?? '-',
                      _eventClock(event),
                    ],
                  )
                  .toList(),
            ),
        ],
      ),
    );
    await Printing.sharePdf(
      bytes: await report.save(),
      filename: 'attendance-$_dateKey.pdf',
    );
  }

  String _eventClock(Map<String, dynamic> event) {
    final time = DateTime.tryParse(event['occurred_at']?.toString() ?? '')?.toLocal();
    if (time == null) return '—';
    final hour = time.hour % 12 == 0 ? 12 : time.hour % 12;
    return '$hour:${time.minute.toString().padLeft(2, '0')} ${time.hour >= 12 ? 'PM' : 'AM'}';
  }

  Widget _filterDropdown({
    required String value,
    required String hint,
    required IconData icon,
    required List<String> options,
    required ValueChanged<String?> onChanged,
  }) {
    return DropdownButtonFormField<String>(
      value: value,
      isExpanded: true,
      dropdownColor: const Color(0xFF07393C),
      iconEnabledColor: Colors.white70,
      style: GoogleFonts.dmSans(color: Colors.white, fontSize: 9),
      decoration: _inputDecoration(hint, icon: icon),
      items: options
          .map(
            (option) => DropdownMenuItem(
              value: option,
              child: Text(
                option == 'All' ? 'All' : option,
                style: GoogleFonts.dmSans(color: Colors.white, fontSize: 9),
              ),
            ),
          )
          .toList(),
      onChanged: onChanged,
    );
  }

  Widget _studentFilters() {
    return _GlassCard(
      padding: const EdgeInsets.all(12),
      child: Column(
        children: [
          TextField(
            controller: _studentSearchController,
            onChanged: (_) => setState(() {}),
            style: GoogleFonts.dmSans(color: Colors.white, fontSize: 10),
            decoration:
                _inputDecoration(
                  'Search by student name',
                  icon: Icons.search_rounded,
                ).copyWith(
                  suffixIcon: _studentSearchController.text.isEmpty
                      ? null
                      : IconButton(
                          onPressed: () {
                            _studentSearchController.clear();
                            setState(() {});
                          },
                          icon: const Icon(Icons.close_rounded, size: 17),
                          color: Colors.white70,
                        ),
                ),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: _filterDropdown(
                  value: _selectedFloor,
                  hint: 'Floor',
                  icon: Icons.layers_outlined,
                  options: _floorOptions,
                  onChanged: (floor) => setState(() {
                    _selectedFloor = floor ?? 'All';
                    _selectedRoom = 'All';
                  }),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _filterDropdown(
                  value: _selectedRoom,
                  hint: 'Room',
                  icon: Icons.meeting_room_outlined,
                  options: _roomOptions,
                  onChanged: (room) =>
                      setState(() => _selectedRoom = room ?? 'All'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    // Attendance is one student per selected day, not one percentage point per
    // IN/OUT movement.  Build the set from the actual QR event list so an IN
    // followed by an OUT still counts as one recorded resident.
    final residentIds = _students
        .map((student) => student['id']?.toString())
        .whereType<String>()
        .toSet();
    final recordedIds = _events
        .map((event) => (event['student_user_id'] ?? event['student_id'] ?? event['user_id'])?.toString())
        .whereType<String>()
        .where(residentIds.contains)
        .toSet();
    final recordedNames = _events
        .map((event) => event['student_name']?.toString().trim().toLowerCase())
        .whereType<String>()
        .where((name) => name.isNotEmpty)
        .toSet();
    final present = _students.where((student) {
      final id = student['id']?.toString();
      final name = student['name']?.toString().trim().toLowerCase() ?? '';
      return recordedIds.contains(id) || recordedNames.contains(name);
    }).length;

    final percentage = _students.isEmpty
        ? 0
        : ((present / _students.length) * 100).round();

    return _WardenPage(
      title: 'Attendance',
      hostelName: widget.hostelName,
      child: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
        child: Column(
          children: [
            _GlassCard(
              child: InkWell(
                onTap: _chooseDate,
                child: Row(
                  children: [
                    Icon(Icons.calendar_month_rounded, color: _mint),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Attendance day & date',
                            style: GoogleFonts.dmSans(
                              color: Colors.white70,
                              fontSize: 9,
                            ),
                          ),
                          Text(
                            '${_selectedDate.day}/${_selectedDate.month}/${_selectedDate.year}',
                            style: GoogleFonts.dmSans(
                              color: Colors.white,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ],
                      ),
                    ),
                    Icon(Icons.edit_calendar_outlined, color: _mint),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
            _GlassCard(
              child: Row(
                children: [
                  _attendanceStat('${_students.length}', 'Students'),
                  _attendanceStat('$percentage%', 'Attendance'),
                ],
              ),
            ),

            const SizedBox(height: 12),
            _GlassCard(
              padding: const EdgeInsets.all(12),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.qr_code_2_rounded, color: _mint, size: 21),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'This is a read-only QR attendance register. Every scanned IN/OUT entry is saved with the student, date, time, floor, room and bed.',
                      style: GoogleFonts.dmSans(
                        color: Colors.white70,
                        fontSize: 9.5,
                        height: 1.35,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            _studentFilters(),
            const SizedBox(height: 14),

            if (_loading)
              const Padding(
                padding: EdgeInsets.all(32),
                child: CircularProgressIndicator(),
              )
            else if (_visibleEvents.isEmpty)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 28),
                child: Text(
                  'No Attendance Record',
                  style: GoogleFonts.dmSans(
                    color: Colors.white70,
                    fontSize: 10,
                  ),
                ),
              )
            else
              ..._visibleEvents.map(_attendanceEventCard),

            const SizedBox(height: 8),

            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: _downloadPdf,
                icon: const Icon(Icons.picture_as_pdf_outlined),
                label: const Text('Download attendance PDF'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: _mint,
                  side: BorderSide(color: _mint.withOpacity(0.5)),
                  padding: const EdgeInsets.symmetric(vertical: 14),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  DateTime? _eventTime(Map<String, dynamic> event) =>
      DateTime.tryParse(event['occurred_at']?.toString() ?? '')?.toLocal();

  String _eventDateTime(Map<String, dynamic> event) {
    final time = _eventTime(event);
    if (time == null) return 'Date and time unavailable';
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    final hour = time.hour % 12 == 0 ? 12 : time.hour % 12;
    return '${time.day.toString().padLeft(2, '0')} ${months[time.month - 1]} ${time.year}  •  $hour:${time.minute.toString().padLeft(2, '0')} ${time.hour >= 12 ? 'PM' : 'AM'}';
  }

  Widget _attendanceEventCard(Map<String, dynamic> event) {
    final isIn = event['direction']?.toString().toLowerCase() == 'in';
    final name = event['student_name']?.toString() ?? 'Student';
    return Padding(
      padding: const EdgeInsets.only(bottom: 9),
      child: _GlassCard(
        padding: const EdgeInsets.all(13),
        borderColor: isIn ? const Color(0x9950D6B6) : const Color(0x995AA8B7),
        child: Row(
          children: [
            Container(
              width: 43, height: 43,
              decoration: BoxDecoration(color: isIn ? _teal : const Color(0xFF256C6E), borderRadius: BorderRadius.circular(13)),
              child: Icon(isIn ? Icons.login_rounded : Icons.logout_rounded, color: Colors.white),
            ),
            const SizedBox(width: 11),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Row(children: [Expanded(child: Text(name, maxLines: 1, overflow: TextOverflow.ellipsis, style: GoogleFonts.dmSans(color: Colors.white, fontWeight: FontWeight.w800, fontSize: 13))), Text(isIn ? 'IN' : 'OUT', style: GoogleFonts.dmSans(color: isIn ? _mint : const Color(0xFF9BD9E4), fontSize: 10, fontWeight: FontWeight.w800))]),
              const SizedBox(height: 3), Text(_eventDateTime(event), style: GoogleFonts.dmSans(color: Colors.white70, fontSize: 10)),
              const SizedBox(height: 6), Text('Floor ${event['floor_label'] ?? '—'}  •  Room ${event['room_number'] ?? '—'}  •  Bed ${event['bed_number'] ?? '—'}', style: GoogleFonts.dmSans(color: Colors.white70, fontSize: 10, fontWeight: FontWeight.w700)),
            ])),
          ],
        ),
      ),
    );
  }

  Widget _attendanceStat(String value, String label) {
    return Expanded(
      child: Column(
        children: [
          Text(
            value,
            style: GoogleFonts.dmSans(
              color: _mint,
              fontWeight: FontWeight.w800,
              fontSize: 15,
            ),
          ),
          Text(
            label,
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(
              color: Colors.white.withOpacity(0.5),
              fontSize: 7.5,
            ),
          ),
        ],
      ),
    );
  }
}

// =============================================================================
// 3. ANNOUNCEMENTS
// =============================================================================

class WardenAnnouncementsScreen extends StatefulWidget {
  final String hostelName;

  const WardenAnnouncementsScreen({super.key, required this.hostelName});

  @override
  State<WardenAnnouncementsScreen> createState() =>
      _WardenAnnouncementsScreenState();
}

class _WardenAnnouncementsScreenState extends State<WardenAnnouncementsScreen> {
  final _titleCtrl = TextEditingController();

  final _bodyCtrl = TextEditingController();

  List<Map<String, dynamic>> _items = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final rows = await Api().getWardenAnnouncements();
      if (!mounted) return;
      setState(() {
        _items = rows
            .whereType<Map>()
            .map((row) => Map<String, dynamic>.from(row))
            .toList();
        _loading = false;
      });
    } catch (error) {
      if (mounted) {
        setState(() => _loading = false);
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
      }
    }
  }

  @override
  void dispose() {
    _titleCtrl.dispose();
    _bodyCtrl.dispose();
    super.dispose();
  }

  Future<void> _post() async {
    final title = _titleCtrl.text.trim();

    final body = _bodyCtrl.text.trim();

    if (title.isEmpty || body.isEmpty) {
      _mobileToast(context, 'Enter title and announcement.', error: true);
      return;
    }

    try {
      final assigned = await Api().getWardenAssignedHostel();
      final hostel = Map<String, dynamic>.from(assigned['hostel'] as Map);
      await Api().postAnnouncement(
        hostelId: int.parse(hostel['id'].toString()),
        title: title,
        body: body,
      );
      if (!mounted) return;
      _titleCtrl.clear();
      _bodyCtrl.clear();
      FocusScope.of(context).unfocus();
      _mobileToast(context, 'Announcement posted successfully.');
      await _load();
    } catch (error) {
      if (mounted)
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
    }
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: 'Announcements',
      hostelName: widget.hostelName,
      child: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
        child: Column(
          children: [
            _GlassCard(
              child: Column(
                children: [
                  TextField(
                    controller: _titleCtrl,
                    style: const TextStyle(color: Colors.white),
                    decoration: _inputDecoration(
                      'Announcement title',
                      icon: Icons.title_rounded,
                    ),
                  ),

                  const SizedBox(height: 10),

                  TextField(
                    controller: _bodyCtrl,
                    maxLines: 4,
                    style: const TextStyle(color: Colors.white),
                    decoration: _inputDecoration(
                      'Write announcement...',
                      icon: Icons.notes_rounded,
                    ),
                  ),

                  const SizedBox(height: 12),

                  _PrimaryButton(
                    text: 'Post Announcement',
                    icon: Icons.campaign_rounded,
                    onPressed: _post,
                  ),
                ],
              ),
            ),

            const SizedBox(height: 14),

            if (_loading)
              const Padding(
                padding: EdgeInsets.all(28),
                child: CircularProgressIndicator(),
              )
            else if (_items.isEmpty)
              _emptyMessage('No announcements posted yet.')
            else
              ..._items.map(
                (item) => Padding(
                  padding: const EdgeInsets.only(bottom: 9),
                  child: _GlassCard(
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Container(
                          width: 42,
                          height: 42,
                          decoration: BoxDecoration(
                            color: _teal.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(13),
                          ),
                          child: Icon(Icons.campaign_outlined, color: _mint),
                        ),

                        const SizedBox(width: 11),

                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                item['title'] ?? '',
                                style: GoogleFonts.dmSans(
                                  color: Colors.white,
                                  fontWeight: FontWeight.w800,
                                  fontSize: 10.5,
                                ),
                              ),

                              const SizedBox(height: 3),

                              Text(
                                item['body'] ?? '',
                                style: GoogleFonts.dmSans(
                                  color: Colors.white.withOpacity(0.6),
                                  fontSize: 8.7,
                                  height: 1.4,
                                ),
                              ),

                              const SizedBox(height: 5),

                              Text(
                                item['created_at']
                                        ?.toString()
                                        .split('T')
                                        .first ??
                                    '',
                                style: GoogleFonts.dmSans(
                                  color: _mint,
                                  fontSize: 8,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// =============================================================================
// 4. PENALTIES
// =============================================================================

class WardenPenaltiesScreen extends StatefulWidget {
  final String hostelName;

  const WardenPenaltiesScreen({super.key, required this.hostelName});

  @override
  State<WardenPenaltiesScreen> createState() => _WardenPenaltiesScreenState();
}

class _WardenPenaltiesScreenState extends State<WardenPenaltiesScreen> {
  List<Map<String, dynamic>> _penalties = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final rows = await Api().getWardenPenalties();
      if (!mounted) return;
      setState(() {
        _penalties = rows
            .whereType<Map>()
            .map((row) => Map<String, dynamic>.from(row))
            .toList();
        _loading = false;
      });
    } catch (error) {
      if (mounted) {
        setState(() => _loading = false);
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
      }
    }
  }

  Future<void> _openAddPenalty() async {
    final result = await Navigator.push<Map<String, dynamic>>(
      context,
      MaterialPageRoute(
        builder: (_) => _AddPenaltyScreen(hostelName: widget.hostelName),
      ),
    );

    if (!mounted || result == null) {
      return;
    }

    _mobileToast(context, 'Penalty assigned successfully.');
    await _load();
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: 'Penalties',
      hostelName: widget.hostelName,
      floatingActionButton: FloatingActionButton.extended(
        backgroundColor: _teal,
        foregroundColor: Colors.white,
        onPressed: _openAddPenalty,
        icon: const Icon(Icons.add_rounded),
        label: const Text('Add Penalty'),
      ),
      child: ListView.separated(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
        itemCount: _loading
            ? 1
            : _penalties.isEmpty
            ? 1
            : _penalties.length,
        separatorBuilder: (_, __) => const SizedBox(height: 9),
        itemBuilder: (_, index) {
          if (_loading)
            return const Padding(
              padding: EdgeInsets.all(28),
              child: Center(child: CircularProgressIndicator()),
            );
          if (_penalties.isEmpty)
            return _emptyMessage('No penalties have been added.');
          final item = _penalties[index];

          final paid = item['status']?.toString() == 'paid';

          return _GlassCard(
            child: Column(
              children: [
                Row(
                  children: [
                    Container(
                      width: 43,
                      height: 43,
                      decoration: BoxDecoration(
                        color: paid
                            ? _green.withOpacity(0.15)
                            : _red.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(13),
                      ),
                      child: Icon(
                        paid ? Icons.check_rounded : Icons.gavel_rounded,
                        color: paid ? _green : _red,
                      ),
                    ),

                    const SizedBox(width: 11),

                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            item['reason'].toString(),
                            style: GoogleFonts.dmSans(
                              color: Colors.white,
                              fontWeight: FontWeight.w800,
                              fontSize: 10.5,
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            '${item['student_name']} · ${_allocationDetail(floor: item['floor_label'], room: item['room_number'], bed: item['bed_number'])}',
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: GoogleFonts.dmSans(
                              color: Colors.white.withOpacity(0.5),
                              fontSize: 8.5,
                            ),
                          ),
                        ],
                      ),
                    ),

                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text(
                          'Rs. ${item['amount']}',
                          style: GoogleFonts.dmSans(
                            color: _mint,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                        const SizedBox(height: 3),
                        Text(
                          paid ? 'PAID' : 'UNPAID',
                          style: GoogleFonts.dmSans(
                            color: paid ? _green : _orange,
                            fontSize: 7.5,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),

                if (!paid) ...[
                  const SizedBox(height: 10),

                  SizedBox(
                    width: double.infinity,
                    child: OutlinedButton.icon(
                      onPressed: () async {
                        try {
                          await Api().markWardenPenaltyPaid(
                            int.parse(item['id'].toString()),
                          );
                          if (!mounted) return;
                          _mobileToast(context, 'Penalty marked as paid.');
                          await _load();
                        } catch (error) {
                          if (mounted)
                            _mobileToast(
                              context,
                              error.toString().replaceFirst('Exception: ', ''),
                              error: true,
                            );
                        }
                      },
                      icon: const Icon(Icons.payments_outlined, size: 16),
                      label: const Text('Mark as Paid'),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: _mint,
                        side: BorderSide(color: _mint.withOpacity(0.35)),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(13),
                        ),
                      ),
                    ),
                  ),
                ],
              ],
            ),
          );
        },
      ),
    );
  }
}

class _AddPenaltyScreen extends StatefulWidget {
  final String hostelName;

  const _AddPenaltyScreen({required this.hostelName});

  @override
  State<_AddPenaltyScreen> createState() => _AddPenaltyScreenState();
}

class _AddPenaltyScreenState extends State<_AddPenaltyScreen> {
  final _nameCtrl = TextEditingController();

  final _roomCtrl = TextEditingController();

  final _reasonCtrl = TextEditingController();

  final _amountCtrl = TextEditingController();

  bool _loadingStudents = true;
  String? _studentsError;
  List<Map<String, dynamic>> _students = [];
  int? _selectedStudentId;
  String? _selectedRoom;

  @override
  void initState() {
    super.initState();
    _loadStudents();
  }

  Future<void> _loadStudents() async {
    setState(() {
      _loadingStudents = true;
      _studentsError = null;
    });
    try {
      final rows = await Api().getWardenStudents();
      if (!mounted) return;
      setState(() {
        _students = rows
            .whereType<Map>()
            .map((row) => Map<String, dynamic>.from(row))
            .where(
              (student) =>
                  student['student_id'] != null && student['name'] != null,
            )
            .toList();
        _loadingStudents = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loadingStudents = false;
        _studentsError = error.toString().replaceFirst('Exception: ', '');
      });
    }
  }

  List<Map<String, dynamic>> get _studentsForSelectedRoom =>
      _selectedRoom == null
      ? _students
      : _students
            .where(
              (student) => '${student['room_number'] ?? ''}' == _selectedRoom,
            )
            .toList();

  List<String> get _roomNumbers =>
      _students
          .map((student) => '${student['room_number'] ?? ''}'.trim())
          .where((room) => room.isNotEmpty)
          .toSet()
          .toList()
        ..sort();

  @override
  void dispose() {
    _nameCtrl.dispose();
    _roomCtrl.dispose();
    _reasonCtrl.dispose();
    _amountCtrl.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final reason = _reasonCtrl.text.trim();
    final amountText = _amountCtrl.text.trim();
    final amount = int.tryParse(amountText) ?? 0;

    if (_selectedStudentId == null ||
        _roomCtrl.text.trim().isEmpty ||
        reason.isEmpty ||
        RegExp(r'\d').hasMatch(reason) ||
        !RegExp(r'[A-Za-z]').hasMatch(reason) ||
        !RegExp(r'^\d+$').hasMatch(amountText) ||
        amount <= 0) {
      _mobileToast(
        context,
        'Enter a text-only reason and a valid numeric amount.',
        error: true,
      );
      return;
    }

    try {
      await Api().assignPenalty(
        studentId: _selectedStudentId!,
        reason: _reasonCtrl.text.trim(),
        amount: amount.toDouble(),
      );
      if (mounted) Navigator.pop(context, <String, dynamic>{'saved': true});
    } catch (error) {
      if (mounted)
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
    }
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: 'Add Penalty',
      hostelName: widget.hostelName,
      child: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
        child: _GlassCard(
          child: Column(
            children: [
              if (_loadingStudents)
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 24),
                  child: CircularProgressIndicator(),
                )
              else if (_studentsError != null)
                Column(
                  children: [
                    Text(
                      _studentsError!,
                      textAlign: TextAlign.center,
                      style: GoogleFonts.dmSans(
                        color: Colors.white70,
                        fontSize: 10,
                      ),
                    ),
                    TextButton.icon(
                      onPressed: _loadStudents,
                      icon: const Icon(Icons.refresh_rounded),
                      label: const Text('Retry'),
                    ),
                  ],
                )
              else ...[
                DropdownButtonFormField<String>(
                  value: _selectedRoom,
                  isExpanded: true,
                  dropdownColor: const Color(0xFF07393C),
                  iconEnabledColor: _mint,
                  style: GoogleFonts.dmSans(color: Colors.white),
                  hint: Text(
                    'Room number',
                    style: GoogleFonts.dmSans(color: Colors.white70),
                  ),
                  decoration: _inputDecoration(
                    'Room number',
                    icon: Icons.meeting_room_outlined,
                  ),
                  items: _roomNumbers
                      .map(
                        (room) => DropdownMenuItem(
                          value: room,
                          child: Text(
                            'Room $room',
                            style: GoogleFonts.dmSans(color: Colors.white),
                          ),
                        ),
                      )
                      .toList(),
                  onChanged: (room) {
                    setState(() {
                      _selectedRoom = room;
                      _roomCtrl.text = room ?? '';
                      final stillMatches = _studentsForSelectedRoom.any(
                        (student) =>
                            int.tryParse(student['student_id'].toString()) ==
                            _selectedStudentId,
                      );
                      if (!stillMatches) {
                        _selectedStudentId = null;
                        _nameCtrl.clear();
                      }
                    });
                  },
                ),

                const SizedBox(height: 10),

                DropdownButtonFormField<int>(
                  value: _selectedStudentId,
                  isExpanded: true,
                  dropdownColor: const Color(0xFF07393C),
                  iconEnabledColor: _mint,
                  style: GoogleFonts.dmSans(color: Colors.white),
                  hint: Text(
                    'Student name',
                    style: GoogleFonts.dmSans(color: Colors.white70),
                  ),
                  decoration: _inputDecoration(
                    'Student name',
                    icon: Icons.person_outline,
                  ),
                  items: _studentsForSelectedRoom
                      .map((student) {
                        final id = int.tryParse('${student['student_id']}');
                        return DropdownMenuItem<int>(
                          value: id,
                          child: Text(
                            '${student['name']}',
                            style: GoogleFonts.dmSans(color: Colors.white),
                          ),
                        );
                      })
                      .whereType<DropdownMenuItem<int>>()
                      .toList(),
                  onChanged: (studentId) {
                    final student = _students.firstWhere(
                      (item) =>
                          int.tryParse(item['student_id'].toString()) ==
                          studentId,
                    );
                    setState(() {
                      _selectedStudentId = studentId;
                      _nameCtrl.text = '${student['name']}';
                      _selectedRoom =
                          '${student['room_number'] ?? ''}'.trim().isEmpty
                          ? null
                          : '${student['room_number']}';
                      _roomCtrl.text = _selectedRoom ?? '';
                    });
                  },
                ),

                const SizedBox(height: 10),
              ],

              TextField(
                controller: _reasonCtrl,
                inputFormatters: [
                  FilteringTextInputFormatter.deny(RegExp(r'\d')),
                ],
                style: const TextStyle(color: Colors.white),
                decoration: _inputDecoration(
                  'Penalty reason',
                  icon: Icons.gavel_outlined,
                ),
              ),

              const SizedBox(height: 10),

              TextField(
                controller: _amountCtrl,
                keyboardType: TextInputType.number,
                inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                style: const TextStyle(color: Colors.white),
                decoration: _inputDecoration(
                  'Amount',
                  icon: Icons.payments_outlined,
                ),
              ),

              const SizedBox(height: 16),

              _PrimaryButton(
                text: 'Assign Penalty',
                icon: Icons.gavel_rounded,
                onPressed: _save,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// 5. ROOM ALLOCATION
// =============================================================================

class WardenRoomAllocationScreen extends StatefulWidget {
  final String hostelName;

  const WardenRoomAllocationScreen({super.key, required this.hostelName});

  @override
  State<WardenRoomAllocationScreen> createState() =>
      _WardenRoomAllocationScreenState();
}

class _WardenRoomAllocationScreenState
    extends State<WardenRoomAllocationScreen> {
  String _action = 'Allocate Room/Bed';
  String? _selectedAllocation;
  String? _targetFloor;
  String? _targetRoom;
  int? _targetBed;
  List<Map<String, dynamic>> _rooms = [];
  bool _loadingRooms = true;

  @override
  void initState() {
    super.initState();
    _loadRooms();
  }

  Future<void> _loadRooms() async {
    try {
      final rows = await Api().getWardenRooms();
      if (!mounted) return;
      setState(() {
        _rooms = rows.whereType<Map>().map((row) {
          final room = Map<String, dynamic>.from(row);
          final residents = List<dynamic>.from(
            room['students'] as List? ?? const [],
          );
          final names = <String>[];
          final bookingIds = <String, int>{};
          final bedNumbers = <String, String>{};
          for (final rawResident in residents.whereType<Map>()) {
            final resident = Map<String, dynamic>.from(rawResident);
            final name = resident['name']?.toString();
            if (name == null || name.isEmpty) continue;
            names.add(name);
            bookingIds[name] =
                int.tryParse(resident['booking_id'].toString()) ?? 0;
            bedNumbers[name] =
                resident['bed_number']?.toString() ?? '${names.length}';
          }
          return {
            'id': room['id'],
            'room': room['room_number'].toString(),
            'floor': room['floor_label'].toString(),
            'total': int.tryParse(room['capacity'].toString()) ?? 0,
            'students': names,
            'bookingIds': bookingIds,
            'bedNumbers': bedNumbers,
          };
        }).toList();
        _loadingRooms = false;
      });
    } catch (error) {
      if (mounted) setState(() => _loadingRooms = false);
    }
  }

  Future<void> _allocate(Map<String, dynamic> room) async {
    final students = List<String>.from(room['students'] as List);

    final total = room['total'] as int;

    if (students.length >= total) {
      _mobileToast(context, 'This room is already full.', error: true);
      return;
    }

    final result = await Navigator.push<String>(
      context,
      MaterialPageRoute(
        builder: (_) => _AllocateStudentScreen(
          hostelName: widget.hostelName,
          roomNumber: room['room'].toString(),
          floorNumber: room['floor'].toString(),
          totalBeds: total,
          occupiedBeds: students.length,
        ),
      ),
    );

    if (!mounted || result == null) {
      return;
    }

    setState(() {
      students.add(result);
      room['students'] = students;
    });

    _mobileToast(
      context,
      '$result allocated to Floor ${room['floor']} · Room ${room['room']}.',
    );
  }

  void _selectAllocatedStudent(String? value) {
    if (value == null) return;
    final parts = value.split('|');
    final room = _rooms.firstWhere(
      (item) => item['room'].toString() == parts.first,
    );
    setState(() {
      _selectedAllocation = value;
      _targetFloor = room['floor'].toString();
      _targetRoom = room['room'].toString();
      _targetBed = 1;
    });
  }

  Future<void> _applyRoomChange({bool deallocate = false}) async {
    if (_selectedAllocation == null) return;
    final parts = _selectedAllocation!.split('|');
    final student = parts.last;
    final selectedBed = _targetBed;
    if (!deallocate &&
        (_targetFloor == null || _targetRoom == null || _targetBed == null))
      return;
    final target = deallocate
        ? null
        : _rooms.firstWhere((room) => room['room'].toString() == _targetRoom);
    if (target != null &&
        List<String>.from(target['students'] as List).length >=
            (target['total'] as int) &&
        parts.first != _targetRoom) {
      _mobileToast(context, 'The selected room is already full.', error: true);
      return;
    }
    final source = _rooms.firstWhere(
      (room) => room['room'].toString() == parts.first,
    );
    final bookingIds = Map<String, int>.from(source['bookingIds'] as Map);
    final bookingId = bookingIds[student];
    final targetRoomId = target == null
        ? null
        : int.tryParse(target['id'].toString());
    if (bookingId == null ||
        bookingId == 0 ||
        (!deallocate && targetRoomId == null)) {
      _mobileToast(
        context,
        'This allocation is missing its saved booking record.',
        error: true,
      );
      return;
    }
    try {
      if (deallocate) {
        await Api().deallocateBookingRoom(bookingId);
      } else {
        await Api().changeBookingAllocation(
          bookingId: bookingId,
          roomId: targetRoomId!,
          bedNumber: selectedBed.toString(),
        );
      }
      if (!mounted) return;
      setState(() {
        _selectedAllocation = null;
        _targetFloor = null;
        _targetRoom = null;
        _targetBed = null;
      });
      await _loadRooms();
      if (!mounted) return;
      _mobileToast(
        context,
        deallocate
            ? '$student has been deallocated. Their previous bed is now free.'
            : '$student moved to Floor ${target!['floor']} · Room ${target['room']} · Bed $selectedBed.',
      );
    } catch (error) {
      if (mounted)
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
    }
  }

  Widget _buildInlineChangeForm() {
    final parts = _selectedAllocation!.split('|');
    final student = parts.last;
    final floorRooms = _rooms
        .where((room) => room['floor'].toString() == _targetFloor)
        .toList();
    final selectedRoom = _rooms.firstWhere(
      (room) => room['room'].toString() == _targetRoom,
    );
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
      child: _GlassCard(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Change allocation for $student',
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontWeight: FontWeight.w800,
                fontSize: 13,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              'Choose the new Floor, Room and Bed, then update.',
              style: GoogleFonts.dmSans(color: Colors.white70, fontSize: 9),
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              value: _targetFloor,
              dropdownColor: const Color(0xFF07393C),
              style: const TextStyle(color: Colors.white),
              decoration: _inputDecoration(
                'Floor',
                icon: Icons.layers_outlined,
              ),
              items: _rooms
                  .map((room) => room['floor'].toString())
                  .toSet()
                  .map(
                    (floor) => DropdownMenuItem(
                      value: floor,
                      child: Text('Floor $floor'),
                    ),
                  )
                  .toList(),
              onChanged: (floor) => setState(() {
                _targetFloor = floor;
                _targetRoom = _rooms
                    .firstWhere(
                      (room) => room['floor'].toString() == floor,
                    )['room']
                    .toString();
                _targetBed = 1;
              }),
            ),
            const SizedBox(height: 9),
            DropdownButtonFormField<String>(
              value: _targetRoom,
              dropdownColor: const Color(0xFF07393C),
              style: const TextStyle(color: Colors.white),
              decoration: _inputDecoration(
                'Room number',
                icon: Icons.meeting_room_outlined,
              ),
              items: floorRooms
                  .map(
                    (room) => DropdownMenuItem(
                      value: room['room'].toString(),
                      child: Text('Room ${room['room']}'),
                    ),
                  )
                  .toList(),
              onChanged: (room) => setState(() {
                _targetRoom = room;
                _targetBed = 1;
              }),
            ),
            const SizedBox(height: 9),
            DropdownButtonFormField<int>(
              value: _targetBed,
              dropdownColor: const Color(0xFF07393C),
              style: const TextStyle(color: Colors.white),
              decoration: _inputDecoration(
                'Bed number',
                icon: Icons.bed_outlined,
              ),
              items: List.generate(
                selectedRoom['total'] as int,
                (index) => DropdownMenuItem(
                  value: index + 1,
                  child: Text('Bed ${index + 1}'),
                ),
              ),
              onChanged: (bed) => setState(() => _targetBed = bed),
            ),
            const SizedBox(height: 13),
            _PrimaryButton(
              text: 'Update Room & Bed',
              icon: Icons.save_rounded,
              onPressed: _applyRoomChange,
            ),
            const SizedBox(height: 8),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: () => _applyRoomChange(deallocate: true),
                icon: const Icon(Icons.person_remove_outlined),
                label: const Text('Deallocate student'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: _red,
                  side: const BorderSide(color: _red),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAllocatedStudentsList() {
    final allocations = <Map<String, dynamic>>[];
    for (final room in _rooms) {
      final students = List<String>.from(room['students'] as List);
      for (var index = 0; index < students.length; index++) {
        allocations.add({
          'name': students[index],
          'floor': room['floor'].toString(),
          'room': room['room'].toString(),
          'bed':
              Map<String, String>.from(
                room['bedNumbers'] as Map,
              )[students[index]] ??
              '${index + 1}',
        });
      }
    }
    return ListView.separated(
      physics: const BouncingScrollPhysics(),
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
      itemCount: allocations.length,
      separatorBuilder: (_, __) => const SizedBox(height: 9),
      itemBuilder: (_, index) {
        final allocation = allocations[index];
        final name = allocation['name'].toString();
        return InkWell(
          onTap: () => _selectAllocatedStudent('${allocation['room']}|$name'),
          borderRadius: BorderRadius.circular(20),
          child: _GlassCard(
            child: Row(
              children: [
                CircleAvatar(
                  radius: 22,
                  backgroundColor: _teal.withOpacity(0.20),
                  child: Text(
                    name[0].toUpperCase(),
                    style: GoogleFonts.dmSans(
                      color: _mint,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ),
                const SizedBox(width: 11),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        name,
                        style: GoogleFonts.dmSans(
                          color: Colors.white,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Floor ${allocation['floor']} · Room ${allocation['room']} · Bed ${allocation['bed']}',
                        style: GoogleFonts.dmSans(color: _mint, fontSize: 9),
                      ),
                      const SizedBox(height: 3),
                      Text(
                        'Tap to change room/bed or deallocate',
                        style: GoogleFonts.dmSans(
                          color: Colors.white.withOpacity(0.5),
                          fontSize: 8.5,
                        ),
                      ),
                    ],
                  ),
                ),
                Icon(Icons.edit_outlined, color: _mint, size: 19),
              ],
            ),
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: 'Room & Beds',
      hostelName: widget.hostelName,
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 8),
            child: DropdownButtonFormField<String>(
              value: _action,
              dropdownColor: const Color(0xFF07393C),
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontWeight: FontWeight.w700,
              ),
              decoration: _inputDecoration(
                'Choose action',
                icon: Icons.tune_rounded,
              ),
              items: const [
                DropdownMenuItem(
                  value: 'Allocate Room/Bed',
                  child: Text('Allocate Room/Bed'),
                ),
                DropdownMenuItem(
                  value: 'Change Room/Bed',
                  child: Text('Change Room/Bed'),
                ),
              ],
              onChanged: (value) => setState(() => _action = value!),
            ),
          ),
          if (_action == 'Change Room/Bed' && _selectedAllocation != null)
            _buildInlineChangeForm(),
          Expanded(
            child: _loadingRooms
                ? const Center(child: CircularProgressIndicator())
                : _action == 'Change Room/Bed'
                ? _buildAllocatedStudentsList()
                : ListView.separated(
                    physics: const BouncingScrollPhysics(),
                    padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
                    itemCount: _rooms.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 9),
                    itemBuilder: (_, index) {
                      final room = _rooms[index];

                      final total = room['total'] as int;

                      final students = List<String>.from(
                        room['students'] as List,
                      );

                      final free = total - students.length;

                      return _GlassCard(
                        child: Column(
                          children: [
                            Row(
                              children: [
                                Container(
                                  width: 44,
                                  height: 44,
                                  decoration: BoxDecoration(
                                    color: _purple.withOpacity(0.15),
                                    borderRadius: BorderRadius.circular(13),
                                  ),
                                  child: const Icon(
                                    Icons.meeting_room_rounded,
                                    color: _purple,
                                  ),
                                ),

                                const SizedBox(width: 11),

                                Expanded(
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        'Room ${room['room']}',
                                        style: GoogleFonts.dmSans(
                                          color: Colors.white,
                                          fontWeight: FontWeight.w800,
                                        ),
                                      ),
                                      const SizedBox(height: 2),
                                      Text(
                                        'Floor ${room['floor']} · ${students.length} / $total occupied',
                                        style: GoogleFonts.dmSans(
                                          color: Colors.white.withOpacity(0.5),
                                          fontSize: 8.5,
                                        ),
                                      ),
                                      Text(
                                        'Beds: ${students.length} / $total occupied',
                                        style: GoogleFonts.dmSans(
                                          color: Colors.white.withOpacity(0.5),
                                          fontSize: 8.5,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),

                                Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 9,
                                    vertical: 5,
                                  ),
                                  decoration: BoxDecoration(
                                    color: free == 0
                                        ? _red.withOpacity(0.15)
                                        : _green.withOpacity(0.15),
                                    borderRadius: BorderRadius.circular(10),
                                  ),
                                  child: Text(
                                    free == 0 ? 'Full' : '$free free',
                                    style: GoogleFonts.dmSans(
                                      color: free == 0 ? _red : _green,
                                      fontSize: 8,
                                      fontWeight: FontWeight.w800,
                                    ),
                                  ),
                                ),
                              ],
                            ),

                            if (students.isNotEmpty) ...[
                              const SizedBox(height: 10),

                              Align(
                                alignment: Alignment.centerLeft,
                                child: Wrap(
                                  spacing: 6,
                                  runSpacing: 6,
                                  children: students
                                      .map(
                                        (name) => InkWell(
                                          onTap: _action == 'Change Room/Bed'
                                              ? () => _selectAllocatedStudent(
                                                  '${room['room']}|$name',
                                                )
                                              : null,
                                          borderRadius: BorderRadius.circular(
                                            10,
                                          ),
                                          child: Container(
                                            padding: const EdgeInsets.symmetric(
                                              horizontal: 8,
                                              vertical: 5,
                                            ),
                                            decoration: BoxDecoration(
                                              color: Colors.white.withOpacity(
                                                0.06,
                                              ),
                                              borderRadius:
                                                  BorderRadius.circular(10),
                                            ),
                                            child: Text(
                                              _action == 'Change Room/Bed'
                                                  ? '$name  ›'
                                                  : name,
                                              style: GoogleFonts.dmSans(
                                                color: _mint,
                                                fontSize: 8,
                                              ),
                                            ),
                                          ),
                                        ),
                                      )
                                      .toList(),
                                ),
                              ),
                            ],

                            if (free > 0) ...[
                              const SizedBox(height: 10),

                              SizedBox(
                                width: double.infinity,
                                child: OutlinedButton.icon(
                                  onPressed: () {
                                    _allocate(room);
                                  },
                                  icon: const Icon(
                                    Icons.person_add_alt_rounded,
                                    size: 17,
                                  ),
                                  label: const Text('Allocate Student'),
                                  style: OutlinedButton.styleFrom(
                                    foregroundColor: _mint,
                                    side: BorderSide(
                                      color: _mint.withOpacity(0.35),
                                    ),
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(13),
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ],
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}

class _ChangeAllocationScreen extends StatefulWidget {
  final String hostelName;
  final String studentName;
  final String currentFloor;
  final String currentRoom;
  final List<Map<String, dynamic>> rooms;
  const _ChangeAllocationScreen({
    required this.hostelName,
    required this.studentName,
    required this.currentFloor,
    required this.currentRoom,
    required this.rooms,
  });
  @override
  State<_ChangeAllocationScreen> createState() =>
      _ChangeAllocationScreenState();
}

class _ChangeAllocationScreenState extends State<_ChangeAllocationScreen> {
  late String _floor;
  late String _room;
  int _bed = 1;
  @override
  void initState() {
    super.initState();
    _floor = widget.currentFloor;
    _room = widget.currentRoom;
  }

  List<Map<String, dynamic>> get _floorRooms =>
      widget.rooms.where((room) => room['floor'].toString() == _floor).toList();
  Map<String, dynamic> get _selectedRoom =>
      widget.rooms.firstWhere((room) => room['room'].toString() == _room);
  @override
  Widget build(BuildContext context) => _WardenPage(
    title: 'Change Room/Bed',
    hostelName: widget.hostelName,
    child: SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 80),
      child: _GlassCard(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              widget.studentName,
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontWeight: FontWeight.w800,
                fontSize: 18,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              'Currently: Floor ${widget.currentFloor} · Room ${widget.currentRoom}',
              style: GoogleFonts.dmSans(color: Colors.white70, fontSize: 10),
            ),
            const SizedBox(height: 18),
            DropdownButtonFormField<String>(
              value: _floor,
              dropdownColor: const Color(0xFF07393C),
              style: const TextStyle(color: Colors.white),
              decoration: _inputDecoration(
                'Floor',
                icon: Icons.layers_outlined,
              ),
              items: widget.rooms
                  .map((r) => r['floor'].toString())
                  .toSet()
                  .map(
                    (f) => DropdownMenuItem(value: f, child: Text('Floor $f')),
                  )
                  .toList(),
              onChanged: (v) => setState(() {
                _floor = v!;
                _room = _floorRooms.first['room'].toString();
                _bed = 1;
              }),
            ),
            const SizedBox(height: 10),
            DropdownButtonFormField<String>(
              value: _room,
              dropdownColor: const Color(0xFF07393C),
              style: const TextStyle(color: Colors.white),
              decoration: _inputDecoration(
                'Room',
                icon: Icons.meeting_room_outlined,
              ),
              items: _floorRooms
                  .map(
                    (r) => DropdownMenuItem(
                      value: r['room'].toString(),
                      child: Text('Room ${r['room']}'),
                    ),
                  )
                  .toList(),
              onChanged: (v) => setState(() {
                _room = v!;
                _bed = 1;
              }),
            ),
            const SizedBox(height: 10),
            DropdownButtonFormField<int>(
              value: _bed,
              dropdownColor: const Color(0xFF07393C),
              style: const TextStyle(color: Colors.white),
              decoration: _inputDecoration(
                'Bed number',
                icon: Icons.bed_outlined,
              ),
              items: List.generate(
                _selectedRoom['total'] as int,
                (i) =>
                    DropdownMenuItem(value: i + 1, child: Text('Bed ${i + 1}')),
              ),
              onChanged: (v) => setState(() => _bed = v!),
            ),
            const SizedBox(height: 18),
            _PrimaryButton(
              text: 'Update Room & Bed',
              icon: Icons.save_rounded,
              onPressed: () => Navigator.pop(context, {
                'floor': _floor,
                'room': _room,
                'bed': _bed,
              }),
            ),
            const SizedBox(height: 10),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: () => Navigator.pop(context, {'deallocated': true}),
                icon: const Icon(Icons.person_remove_outlined),
                label: const Text('Deallocate student'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: _red,
                  side: const BorderSide(color: _red),
                ),
              ),
            ),
          ],
        ),
      ),
    ),
  );
}

class _AllocateStudentScreen extends StatefulWidget {
  final String hostelName;
  final String roomNumber;
  final String floorNumber;
  final int totalBeds;
  final int occupiedBeds;

  const _AllocateStudentScreen({
    required this.hostelName,
    required this.roomNumber,
    required this.floorNumber,
    required this.totalBeds,
    required this.occupiedBeds,
  });

  @override
  State<_AllocateStudentScreen> createState() => _AllocateStudentScreenState();
}

class _AllocateStudentScreenState extends State<_AllocateStudentScreen> {
  final _studentCtrl = TextEditingController();

  bool _loadingStudents = true;
  String? _studentsError;
  List<Map<String, dynamic>> _students = [];
  int? _selectedStudentId;

  late int _bed;

  @override
  void initState() {
    super.initState();
    _bed = widget.occupiedBeds + 1;
    _loadStudents();
  }

  Future<void> _loadStudents() async {
    setState(() {
      _loadingStudents = true;
      _studentsError = null;
    });
    try {
      final rows = await Api().getWardenStudents();
      if (!mounted) return;
      setState(() {
        _students = rows
            .whereType<Map>()
            .map((row) => Map<String, dynamic>.from(row))
            .where(
              (student) => student['id'] != null && student['name'] != null,
            )
            .toList();
        _loadingStudents = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loadingStudents = false;
        _studentsError = error.toString().replaceFirst('Exception: ', '');
      });
    }
  }

  @override
  void dispose() {
    _studentCtrl.dispose();
    super.dispose();
  }

  void _save() {
    if (_selectedStudentId == null || _studentCtrl.text.trim().isEmpty) {
      _mobileToast(context, 'Select a student.', error: true);
      return;
    }

    Navigator.pop(context, _studentCtrl.text.trim());
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: 'Allocate • Floor ${widget.floorNumber}',
      hostelName: widget.hostelName,
      child: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
        child: _GlassCard(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Floor ${widget.floorNumber} · Room ${widget.roomNumber} · Bed $_bed',
                style: GoogleFonts.dmSans(
                  color: _mint,
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 12),
              if (_loadingStudents)
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 22),
                  child: Center(child: CircularProgressIndicator()),
                )
              else if (_studentsError != null)
                Column(
                  children: [
                    Text(
                      _studentsError!,
                      textAlign: TextAlign.center,
                      style: GoogleFonts.dmSans(
                        color: Colors.white70,
                        fontSize: 10,
                      ),
                    ),
                    TextButton.icon(
                      onPressed: _loadStudents,
                      icon: const Icon(Icons.refresh_rounded),
                      label: const Text('Retry'),
                    ),
                  ],
                )
              else
                DropdownButtonFormField<int>(
                  value: _selectedStudentId,
                  isExpanded: true,
                  dropdownColor: const Color(0xFF07393C),
                  iconEnabledColor: _mint,
                  style: GoogleFonts.dmSans(color: Colors.white),
                  hint: Text(
                    'Student name',
                    style: GoogleFonts.dmSans(color: Colors.white70),
                  ),
                  decoration: _inputDecoration(
                    'Student name',
                    icon: Icons.person_outline,
                  ),
                  items: _students.map((student) {
                    final id = int.tryParse('${student['id']}');
                    return DropdownMenuItem<int>(
                      value: id,
                      child: Text(
                        '${student['name']}',
                        style: GoogleFonts.dmSans(color: Colors.white),
                      ),
                    );
                  }).toList(),
                  onChanged: (studentId) {
                    final student = _students.firstWhere(
                      (item) => int.tryParse('${item['id']}') == studentId,
                    );
                    setState(() {
                      _selectedStudentId = studentId;
                      _studentCtrl.text = '${student['name']}';
                    });
                  },
                ),

              const SizedBox(height: 12),

              DropdownButtonFormField<int>(
                value: _bed,
                dropdownColor: const Color(0xFF082F31),
                style: GoogleFonts.dmSans(color: Colors.white),
                decoration: _inputDecoration(
                  'Select bed',
                  icon: Icons.bed_outlined,
                ),
                items: List.generate(widget.totalBeds - widget.occupiedBeds, (
                  index,
                ) {
                  final bed = widget.occupiedBeds + index + 1;

                  return DropdownMenuItem(value: bed, child: Text('Bed $bed'));
                }),
                onChanged: (value) {
                  if (value != null) {
                    setState(() {
                      _bed = value;
                    });
                  }
                },
              ),

              const SizedBox(height: 16),

              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: _teal.withOpacity(0.10),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Text(
                  'The warden can manually allocate a room and bed to a student who has not booked through the student app.',
                  style: GoogleFonts.dmSans(
                    color: Colors.white.withOpacity(0.68),
                    fontSize: 9,
                    height: 1.45,
                  ),
                ),
              ),

              const SizedBox(height: 16),

              _PrimaryButton(
                text: 'Allocate Room & Bed',
                icon: Icons.meeting_room_rounded,
                onPressed: _save,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// 6. STUDENTS & PARENTS
// =============================================================================

class WardenStudentDetailsScreen extends StatefulWidget {
  final String hostelName;

  const WardenStudentDetailsScreen({super.key, required this.hostelName});

  @override
  State<WardenStudentDetailsScreen> createState() =>
      _WardenStudentDetailsScreenState();
}

class _WardenStudentDetailsScreenState
    extends State<WardenStudentDetailsScreen> {
  List<Map<String, dynamic>> _students = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadStudents();
  }

  Future<void> _loadStudents() async {
    try {
      final rows = await Api().getWardenStudents();
      if (!mounted) return;
      setState(() {
        _students = rows
            .whereType<Map>()
            .map((row) => Map<String, dynamic>.from(row))
            .toList();
        _loading = false;
      });
    } catch (error) {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _openDetails(Map<String, dynamic> student) async {
    final result = await Navigator.push<bool>(
      context,
      MaterialPageRoute(
        builder: (_) => WardenStudentEditScreen(
          hostelName: widget.hostelName,
          student: Map<String, dynamic>.from(student),
        ),
      ),
    );

    if (!mounted || result != true) {
      return;
    }
    await _loadStudents();
  }

  Future<void> _addStudent() async {
    final saved = await Navigator.push<bool>(
      context,
      MaterialPageRoute(
        builder: (_) => WardenStudentEditScreen(
          hostelName: widget.hostelName,
          student: const {},
          isNew: true,
        ),
      ),
    );
    if (saved == true && mounted) await _loadStudents();
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: 'Students & Parents',
      hostelName: widget.hostelName,
      child: _loading
          ? const Center(child: CircularProgressIndicator())
          : ListView.separated(
              physics: const BouncingScrollPhysics(),
              padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
              itemCount: _students.length + 1,
              separatorBuilder: (_, __) => const SizedBox(height: 9),
              itemBuilder: (_, index) {
                if (index == 0) {
                  return _PrimaryButton(
                    text: 'Add New Student',
                    icon: Icons.person_add_alt_1_rounded,
                    onPressed: _addStudent,
                  );
                }
                final student = _students[index - 1];

                final name = student['name'] ?? '';

                return InkWell(
                  onTap: () {
                    _openDetails(student);
                  },
                  borderRadius: BorderRadius.circular(20),
                  child: _GlassCard(
                    child: Row(
                      children: [
                        CircleAvatar(
                          radius: 22,
                          backgroundColor: _teal.withOpacity(0.20),
                          child: Text(
                            name.isEmpty ? 'S' : name[0].toUpperCase(),
                            style: GoogleFonts.dmSans(
                              color: _mint,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ),

                        const SizedBox(width: 11),

                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                name,
                                style: GoogleFonts.dmSans(
                                  color: Colors.white,
                                  fontWeight: FontWeight.w800,
                                ),
                              ),
                              const SizedBox(height: 2),
                              Text(
                                _allocationDetail(
                                  floor: student['floor_label'],
                                  room: student['room_number'],
                                  bed: student['bed_number'],
                                ),
                                style: GoogleFonts.dmSans(
                                  color: _mint,
                                  fontSize: 8.5,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                student['guardian_name'] == null ||
                                        student['guardian_name']
                                            .toString()
                                            .isEmpty
                                    ? 'Guardian details not added'
                                    : 'Guardian: ${student['guardian_name']}',
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: GoogleFonts.dmSans(
                                  color: Colors.white.withOpacity(0.5),
                                  fontSize: 8.5,
                                ),
                              ),
                            ],
                          ),
                        ),

                        Column(
                          children: [
                            Icon(Icons.edit_outlined, color: _mint, size: 18),
                            const SizedBox(height: 3),
                            Text(
                              'View / Edit',
                              style: GoogleFonts.dmSans(
                                color: _mint,
                                fontSize: 6.8,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
    );
  }
}

class WardenStudentEditScreen extends StatefulWidget {
  final String hostelName;
  final Map<String, dynamic> student;
  final bool isNew;

  const WardenStudentEditScreen({
    super.key,
    required this.hostelName,
    required this.student,
    this.isNew = false,
  });

  @override
  State<WardenStudentEditScreen> createState() =>
      _WardenStudentEditScreenState();
}

class _WardenStudentEditScreenState extends State<WardenStudentEditScreen> {
  late final TextEditingController _nameCtrl;

  late final TextEditingController _roomCtrl;

  late final TextEditingController _phoneCtrl;

  late final TextEditingController _parentCtrl;

  late final TextEditingController _parentPhoneCtrl;
  late final TextEditingController _emailCtrl;
  late final TextEditingController _cnicCtrl;
  late final TextEditingController _addressCtrl;
  late final TextEditingController _relationshipCtrl;
  late final TextEditingController _parentEmailCtrl;
  late final TextEditingController _parentAddressCtrl;
  bool _saving = false;

  @override
  void initState() {
    super.initState();

    _nameCtrl = TextEditingController(text: widget.student['name'] ?? '');

    _roomCtrl = TextEditingController(text: widget.student['room'] ?? '');

    _phoneCtrl = TextEditingController(text: widget.student['phone'] ?? '');

    _parentCtrl = TextEditingController(
      text: widget.student['guardian_name']?.toString() ?? '',
    );

    _parentPhoneCtrl = TextEditingController(
      text: widget.student['guardian_phone']?.toString() ?? '',
    );
    _emailCtrl = TextEditingController(
      text: widget.student['email']?.toString() ?? '',
    );
    _cnicCtrl = TextEditingController(
      text: widget.student['cnic']?.toString() ?? '',
    );
    _addressCtrl = TextEditingController(
      text: widget.student['address']?.toString() ?? '',
    );
    _relationshipCtrl = TextEditingController(
      text: widget.student['guardian_relationship']?.toString() ?? '',
    );
    _parentEmailCtrl = TextEditingController(
      text: widget.student['guardian_email']?.toString() ?? '',
    );
    _parentAddressCtrl = TextEditingController(
      text: widget.student['guardian_address']?.toString() ?? '',
    );
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _roomCtrl.dispose();
    _phoneCtrl.dispose();
    _parentCtrl.dispose();
    _parentPhoneCtrl.dispose();
    _emailCtrl.dispose();
    _cnicCtrl.dispose();
    _addressCtrl.dispose();
    _relationshipCtrl.dispose();
    _parentEmailCtrl.dispose();
    _parentAddressCtrl.dispose();

    super.dispose();
  }

  Future<void> _save() async {
    if (_nameCtrl.text.trim().isEmpty || _emailCtrl.text.trim().isEmpty) {
      _mobileToast(
        context,
        'Student full name and email are required.',
        error: true,
      );
      return;
    }
    setState(() => _saving = true);
    final payload = <String, dynamic>{
      'name': _nameCtrl.text.trim(),
      'email': _emailCtrl.text.trim(),
      'phone': _phoneCtrl.text.trim(),
      'cnic': _cnicCtrl.text.trim(),
      'address': _addressCtrl.text.trim(),
      'guardian_name': _parentCtrl.text.trim(),
      'guardian_relationship': _relationshipCtrl.text.trim(),
      'guardian_phone': _parentPhoneCtrl.text.trim(),
      'guardian_email': _parentEmailCtrl.text.trim(),
      'guardian_address': _parentAddressCtrl.text.trim(),
    };
    try {
      if (widget.isNew) {
        final created = await Api().createWardenStudent(payload);
        if (!mounted) return;
        final temp = created['temporary_password']?.toString() ?? '';
        _mobileToast(
          context,
          temp.isEmpty
              ? 'Student record created.'
              : 'Student created. Temporary password: $temp',
        );
      } else {
        final id = int.tryParse(widget.student['id'].toString());
        if (id == null) throw Exception('Student ID is missing');
        await Api().updateWardenStudent(id, payload);
      }
      if (mounted) Navigator.pop(context, true);
    } catch (error) {
      if (mounted)
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: widget.isNew
          ? 'Add Student & Guardian'
          : 'Student & Guardian Details',
      hostelName: widget.hostelName,
      child: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
        child: _GlassCard(
          child: Column(
            children: [
              _editField(
                controller: _nameCtrl,
                label: 'Student full name',
                icon: Icons.person_outline,
              ),

              const SizedBox(height: 10),

              _editField(
                controller: _emailCtrl,
                label: 'Student email',
                icon: Icons.email_outlined,
              ),

              const SizedBox(height: 10),

              _editField(
                controller: _phoneCtrl,
                label: 'Student Phone',
                icon: Icons.phone_outlined,
              ),

              const SizedBox(height: 10),

              _editField(
                controller: _cnicCtrl,
                label: 'Student CNIC',
                icon: Icons.badge_outlined,
              ),

              const SizedBox(height: 10),

              _editField(
                controller: _addressCtrl,
                label: 'Student address',
                icon: Icons.home_outlined,
                maxLines: 2,
              ),

              const SizedBox(height: 16),

              Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  'Parent / Guardian Information',
                  style: GoogleFonts.dmSans(
                    color: _mint,
                    fontWeight: FontWeight.w700,
                    fontSize: 11,
                  ),
                ),
              ),

              const SizedBox(height: 10),

              _editField(
                controller: _parentCtrl,
                label: 'Parent Name',
                icon: Icons.family_restroom_rounded,
              ),

              const SizedBox(height: 10),

              _editField(
                controller: _relationshipCtrl,
                label: 'Relationship (e.g. Mother)',
                icon: Icons.family_restroom_rounded,
              ),

              const SizedBox(height: 10),

              _editField(
                controller: _parentPhoneCtrl,
                label: 'Parent Phone',
                icon: Icons.phone_outlined,
              ),

              const SizedBox(height: 10),

              _editField(
                controller: _parentEmailCtrl,
                label: 'Guardian email',
                icon: Icons.email_outlined,
              ),

              const SizedBox(height: 10),

              _editField(
                controller: _parentAddressCtrl,
                label: 'Guardian address',
                icon: Icons.home_outlined,
                maxLines: 2,
              ),

              const SizedBox(height: 16),

              _PrimaryButton(
                text: _saving
                    ? 'Saving...'
                    : (widget.isNew ? 'Create Student Record' : 'Save Changes'),
                icon: Icons.save_rounded,
                onPressed: _saving ? null : _save,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _editField({
    required TextEditingController controller,
    required String label,
    required IconData icon,
    int maxLines = 1,
  }) {
    return TextField(
      controller: controller,
      maxLines: maxLines,
      style: GoogleFonts.dmSans(color: Colors.white, fontSize: 11),
      decoration: _inputDecoration(label, icon: icon),
    );
  }
}

// =============================================================================
// 7. FEES & RENT
// =============================================================================

class WardenFeesScreen extends StatefulWidget {
  final String hostelName;

  const WardenFeesScreen({super.key, required this.hostelName});

  @override
  State<WardenFeesScreen> createState() => _WardenFeesScreenState();
}

class _WardenFeesScreenState extends State<WardenFeesScreen> {
  final _searchController = TextEditingController();
  List<Map<String, dynamic>> _records = [];
  bool _loading = true;
  String? _loadError;
  String _filter = 'All';
  DateTime _selectedMonth = DateTime.now();

  String get _billingMonth {
    return '${_selectedMonth.year.toString().padLeft(4, '0')}-${_selectedMonth.month.toString().padLeft(2, '0')}-01';
  }

  String get _monthLabel {
    const months = [
      'January',
      'February',
      'March',
      'April',
      'May',
      'June',
      'July',
      'August',
      'September',
      'October',
      'November',
      'December',
    ];
    return '${months[_selectedMonth.month - 1]} ${_selectedMonth.year}';
  }

  Future<void> _pickMonth() async {
    DateTime draft = _selectedMonth;
    final picked = await showDialog<DateTime>(
      context: context,
      useRootNavigator: false,
      builder: (dialogContext) => StatefulBuilder(
        builder: (dialogContext, setDialogState) => Dialog(
          insetPadding: const EdgeInsets.symmetric(
            horizontal: 24,
            vertical: 24,
          ),
          backgroundColor: const Color(0xFFF2F5F5),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(24),
          ),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 340),
            child: Padding(
              padding: const EdgeInsets.fromLTRB(12, 18, 12, 10),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Align(
                    alignment: Alignment.centerLeft,
                    child: Padding(
                      padding: EdgeInsets.symmetric(horizontal: 10),
                      child: Text('Select rent month'),
                    ),
                  ),
                  CalendarDatePicker(
                    initialDate: draft,
                    firstDate: DateTime(2020),
                    lastDate: DateTime(DateTime.now().year + 10),
                    onDateChanged: (value) =>
                        setDialogState(() => draft = value),
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      TextButton(
                        onPressed: () => Navigator.pop(dialogContext),
                        child: const Text('Cancel'),
                      ),
                      TextButton(
                        onPressed: () => Navigator.pop(dialogContext, draft),
                        child: const Text('Done'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
    if (picked == null) return;
    setState(() => _selectedMonth = DateTime(picked.year, picked.month));
    await _loadRecords();
  }

  @override
  void initState() {
    super.initState();
    _loadRecords();
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _loadRecords() async {
    if (mounted)
      setState(() {
        _loading = true;
        _loadError = null;
      });
    try {
      final rows = await Api().getWardenFeeRecords(_billingMonth);
      if (!mounted) return;
      setState(() {
        _records = rows
            .whereType<Map>()
            .map((row) => Map<String, dynamic>.from(row))
            .toList();
        _loading = false;
        _loadError = null;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _loadError = error.toString().replaceFirst('Exception: ', '');
      });
      _mobileToast(
        context,
        error.toString().replaceFirst('Exception: ', ''),
        error: true,
      );
    }
  }

  double _amount(dynamic value) => double.tryParse('${value ?? 0}') ?? 0;

  String _money(dynamic value) => 'Rs. ${_amount(value).toStringAsFixed(0)}';

  List<Map<String, dynamic>> get _visibleRecords {
    final search = _searchController.text.trim().toLowerCase();
    return _records.where((record) {
      final due = _amount(record['remaining_amount']);
      final paid = record['invoice_id'] != null && due <= 0;
      final matchesFilter =
          _filter == 'All' ||
          (_filter == 'Paid' && paid) ||
          (_filter == 'Due' && !paid);
      final name = record['student_name']?.toString().toLowerCase() ?? '';
      return matchesFilter && (search.isEmpty || name.contains(search));
    }).toList();
  }

  Future<void> _showCashForm(Map<String, dynamic> record) async {
    final invoiceId = int.tryParse('${record['invoice_id']}');
    if (invoiceId == null) {
      _mobileToast(
        context,
        'A rent invoice must be set up before recording cash.',
        error: true,
      );
      return;
    }
    final existingPaymentId = int.tryParse(
      '${record['latest_cash_payment_id']}',
    );
    final amountController = TextEditingController(
      text: _amount(
        existingPaymentId == null
            ? record['remaining_amount']
            : record['latest_cash_amount'],
      ).toStringAsFixed(0),
    );
    final noteController = TextEditingController(
      text: record['latest_cash_note']?.toString() ?? '',
    );
    final today = DateTime.now();
    final receivedOn =
        '${today.year.toString().padLeft(4, '0')}-${today.month.toString().padLeft(2, '0')}-${today.day.toString().padLeft(2, '0')}';
    final submitted = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: const Color(0xFF07393C),
        title: Text(
          existingPaymentId == null ? 'Record cash rent' : 'Edit cash rent',
          style: GoogleFonts.dmSans(
            color: Colors.white,
            fontWeight: FontWeight.w800,
          ),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '${record['student_name']} · ${_money(record['remaining_amount'])} remaining',
              style: GoogleFonts.dmSans(color: Colors.white70, fontSize: 10),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: amountController,
              keyboardType: const TextInputType.numberWithOptions(
                decimal: true,
              ),
              style: const TextStyle(color: Colors.white),
              decoration: _inputDecoration(
                'Cash amount (PKR)',
                icon: Icons.payments_outlined,
              ),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: noteController,
              maxLines: 2,
              style: const TextStyle(color: Colors.white),
              decoration: _inputDecoration(
                'Note / receipt book reference',
                icon: Icons.note_outlined,
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () async {
              final amount = double.tryParse(amountController.text.trim());
              if (amount == null || amount <= 0) {
                _mobileToast(
                  context,
                  'Enter a valid cash amount.',
                  error: true,
                );
                return;
              }
              try {
                if (existingPaymentId == null) {
                  await Api().recordWardenCashFeePayment(
                    invoiceId: invoiceId,
                    amount: amount,
                    receivedOn: receivedOn,
                    note: noteController.text,
                  );
                } else {
                  await Api().updateWardenCashFeePayment(
                    paymentId: existingPaymentId,
                    amount: amount,
                    receivedOn:
                        record['latest_cash_received_on']
                            ?.toString()
                            .split('T')
                            .first ??
                        receivedOn,
                    note: noteController.text,
                  );
                }
                if (dialogContext.mounted) Navigator.pop(dialogContext, true);
              } catch (error) {
                if (mounted)
                  _mobileToast(
                    context,
                    error.toString().replaceFirst('Exception: ', ''),
                    error: true,
                  );
              }
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: _teal,
              foregroundColor: Colors.white,
            ),
            child: Text(
              existingPaymentId == null ? 'Mark received' : 'Save changes',
            ),
          ),
        ],
      ),
    );
    amountController.dispose();
    noteController.dispose();
    if (submitted == true) {
      await _loadRecords();
      if (mounted) _mobileToast(context, 'Cash rent record saved.');
    }
  }

  Future<void> _setupRentRecord(Map<String, dynamic> record) async {
    final bookingId = int.tryParse('${record['booking_id']}');
    if (bookingId == null) {
      _mobileToast(context, 'Student allocation was not found.', error: true);
      return;
    }
    try {
      await Api().setupWardenRentRecord(
        bookingId: bookingId,
        billingMonth: _billingMonth,
      );
      await _loadRecords();
      if (mounted) {
        _mobileToast(
          context,
          'Monthly rent record is ready. You can now mark cash received.',
        );
      }
    } catch (error) {
      if (mounted) {
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
      }
    }
  }

  Future<void> _viewReceipt(Map<String, dynamic> record) async {
    final receiptId = int.tryParse('${record['receipt_id']}');
    if (receiptId == null) return;
    try {
      final response = await Api().downloadFeeReceipt(receiptId);
      final type =
          response.headers['content-type'] ??
          record['receipt_content_type']?.toString() ??
          '';
      if (!mounted) return;
      if (type.startsWith('image/')) {
        await showDialog<void>(
          context: context,
          builder: (_) => Dialog(
            child: InteractiveViewer(child: Image.memory(response.bodyBytes)),
          ),
        );
      } else {
        await showDialog<void>(
          context: context,
          builder: (_) => Dialog(
            child: SizedBox(
              width: 700,
              height: 700,
              child: PdfPreview(build: (_) async => response.bodyBytes),
            ),
          ),
        );
      }
    } catch (error) {
      if (mounted)
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
    }
  }

  Future<void> _verifyReceipt(Map<String, dynamic> record) async {
    final receiptId = int.tryParse('${record['receipt_id']}');
    if (receiptId == null) return;
    try {
      await Api().reviewWardenFeeReceipt(
        receiptId: receiptId,
        status: 'verified',
        note: 'Verified by warden after receipt review.',
      );
      await _loadRecords();
      if (mounted)
        _mobileToast(context, 'Receipt verified and rent balance updated.');
    } catch (error) {
      if (mounted)
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
    }
  }

  @override
  Widget build(BuildContext context) {
    final paid = _records
        .where(
          (record) =>
              record['invoice_id'] != null &&
              _amount(record['remaining_amount']) <= 0,
        )
        .length;
    final outstanding = _records.fold<double>(
      0,
      (total, record) => total + _amount(record['remaining_amount']),
    );
    return _WardenPage(
      title: 'Fees & Rent',
      hostelName: widget.hostelName,
      child: RefreshIndicator(
        onRefresh: _loadRecords,
        child: ListView(
          physics: const AlwaysScrollableScrollPhysics(
            parent: BouncingScrollPhysics(),
          ),
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
          children: [
            _GlassCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          'Rent records',
                          style: GoogleFonts.dmSans(
                            color: Colors.white,
                            fontWeight: FontWeight.w800,
                            fontSize: 13,
                          ),
                        ),
                      ),
                      TextButton.icon(
                        onPressed: _pickMonth,
                        icon: const Icon(
                          Icons.calendar_month_outlined,
                          size: 15,
                        ),
                        label: Text(_monthLabel),
                        style: TextButton.styleFrom(foregroundColor: _mint),
                      ),
                    ],
                  ),
                  const SizedBox(height: 9),
                  Row(
                    children: [
                      _metric('${_records.length}', 'Students'),
                      _metric('$paid', 'Paid'),
                      _metric(_money(outstanding), 'Outstanding'),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _searchController,
              onChanged: (_) => setState(() {}),
              style: const TextStyle(color: Colors.white),
              decoration: _inputDecoration(
                'Search registered student',
                icon: Icons.search_rounded,
              ),
            ),
            const SizedBox(height: 10),
            Row(
              children: ['All', 'Paid', 'Due']
                  .map(
                    (filter) => Expanded(
                      child: Padding(
                        padding: EdgeInsets.only(
                          right: filter == 'Due' ? 0 : 7,
                        ),
                        child: _filterButton(filter),
                      ),
                    ),
                  )
                  .toList(),
            ),
            const SizedBox(height: 12),
            if (_loading)
              const Padding(
                padding: EdgeInsets.all(32),
                child: Center(child: CircularProgressIndicator()),
              )
            else if (_loadError != null)
              Padding(
                padding: const EdgeInsets.all(28),
                child: Column(
                  children: [
                    Text(
                      'Rent records could not load.',
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Apply the latest database migration, restart the backend, then try again.',
                      textAlign: TextAlign.center,
                      style: GoogleFonts.dmSans(
                        color: Colors.white60,
                        fontSize: 10,
                      ),
                    ),
                    TextButton(
                      onPressed: _loadRecords,
                      child: const Text('Retry'),
                    ),
                  ],
                ),
              )
            else if (_visibleRecords.isEmpty)
              Padding(
                padding: const EdgeInsets.all(28),
                child: Center(
                  child: Text(
                    'No registered student rent records found.',
                    style: GoogleFonts.dmSans(color: Colors.white70),
                  ),
                ),
              )
            else
              ..._visibleRecords.map(_recordCard),
          ],
        ),
      ),
    );
  }

  Widget _metric(String value, String label) => Expanded(
    child: Column(
      children: [
        Text(
          value,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: GoogleFonts.dmSans(
            color: _mint,
            fontWeight: FontWeight.w800,
            fontSize: 12,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          label,
          style: GoogleFonts.dmSans(color: Colors.white60, fontSize: 8),
        ),
      ],
    ),
  );

  Widget _filterButton(String filter) {
    final selected = _filter == filter;
    return InkWell(
      onTap: () => setState(() => _filter = filter),
      borderRadius: BorderRadius.circular(12),
      child: Container(
        height: 36,
        decoration: BoxDecoration(
          color: selected
              ? _teal.withOpacity(0.28)
              : Colors.black.withOpacity(0.2),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: selected
                ? _mint.withOpacity(0.5)
                : Colors.white.withOpacity(0.08),
          ),
        ),
        child: Center(
          child: Text(
            filter,
            style: GoogleFonts.dmSans(
              color: selected ? _mint : Colors.white70,
              fontSize: 9,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
      ),
    );
  }

  Widget _recordCard(Map<String, dynamic> record) {
    final hasInvoice = record['invoice_id'] != null;
    final remaining = _amount(record['remaining_amount']);
    final paid = hasInvoice && remaining <= 0;
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: InkWell(
        onTap: () async {
          final studentId = int.tryParse('${record['student_id']}');
          if (studentId == null) return;
          await Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => WardenStudentRentDetailScreen(
                hostelName: widget.hostelName,
                studentId: studentId,
                studentName: record['student_name']?.toString() ?? 'Student',
                bookingId: int.tryParse('${record['booking_id']}'),
                floor: record['floor_label'],
                room: record['room_number'],
                bed: record['bed_number'],
              ),
            ),
          );
          if (mounted) await _loadRecords();
        },
        borderRadius: BorderRadius.circular(20),
        child: _GlassCard(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  CircleAvatar(
                    radius: 21,
                    backgroundColor: _teal.withOpacity(0.2),
                    child: Text(
                      (record['student_name']?.toString().isNotEmpty ?? false)
                          ? record['student_name'].toString()[0].toUpperCase()
                          : 'S',
                      style: GoogleFonts.dmSans(
                        color: _mint,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ),
                  const SizedBox(width: 11),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          record['student_name']?.toString() ?? 'Student',
                          style: GoogleFonts.dmSans(
                            color: Colors.white,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                        const SizedBox(height: 3),
                        Text(
                          _allocationDetail(
                            floor: record['floor_label'],
                            room: record['room_number'],
                            bed: record['bed_number'],
                          ),
                          style: GoogleFonts.dmSans(
                            color: Colors.white60,
                            fontSize: 8.5,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Column(
                    children: [
                      _feeStatusChip(
                        !hasInvoice
                            ? 'Setup needed'
                            : paid
                            ? 'Paid'
                            : 'Due',
                      ),
                      const SizedBox(height: 5),
                      const Icon(
                        Icons.chevron_right_rounded,
                        color: Colors.white70,
                      ),
                    ],
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: _amountLabel(
                      'Monthly rent',
                      _money(
                        hasInvoice
                            ? record['invoice_amount']
                            : record['room_monthly_rent'],
                      ),
                    ),
                  ),
                  Expanded(
                    child: _amountLabel('Paid', _money(record['paid_amount'])),
                  ),
                  Expanded(child: _amountLabel('Balance', _money(remaining))),
                ],
              ),
              const SizedBox(height: 9),
              Text(
                'Tap to view full rent history and payment details.',
                style: GoogleFonts.dmSans(color: _mint, fontSize: 8.5),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _amountLabel(String label, String amount) => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Text(
        label,
        style: GoogleFonts.dmSans(color: Colors.white60, fontSize: 7.5),
      ),
      const SizedBox(height: 2),
      Text(
        amount,
        style: GoogleFonts.dmSans(
          color: Colors.white,
          fontWeight: FontWeight.w700,
          fontSize: 9,
        ),
      ),
    ],
  );

  Widget _feeStatusChip(String text) {
    final isPaid = text == 'Paid';
    final isSetup = text == 'Setup needed';
    final color = isPaid
        ? _mint
        : isSetup
        ? Colors.white54
        : _orange;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.14),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(
        text,
        style: GoogleFonts.dmSans(
          color: color,
          fontSize: 8,
          fontWeight: FontWeight.w800,
        ),
      ),
    );
  }
}

// =============================================================================
// 8. EMPTY BEDS
// =============================================================================

class WardenStudentRentDetailScreen extends StatefulWidget {
  final String hostelName;
  final int studentId;
  final String studentName;
  final int? bookingId;
  final dynamic floor;
  final dynamic room;
  final dynamic bed;

  const WardenStudentRentDetailScreen({
    super.key,
    required this.hostelName,
    required this.studentId,
    required this.studentName,
    required this.bookingId,
    required this.floor,
    required this.room,
    required this.bed,
  });

  @override
  State<WardenStudentRentDetailScreen> createState() =>
      _WardenStudentRentDetailScreenState();
}

class _WardenStudentRentDetailScreenState
    extends State<WardenStudentRentDetailScreen> {
  List<Map<String, dynamic>> _records = [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  double _amount(dynamic value) => double.tryParse('${value ?? 0}') ?? 0;
  String _money(dynamic value) => 'Rs. ${_amount(value).toStringAsFixed(0)}';
  String _isoDate(DateTime value) =>
      '${value.year.toString().padLeft(4, '0')}-${value.month.toString().padLeft(2, '0')}-${value.day.toString().padLeft(2, '0')}';

  String _monthLabel(dynamic value) {
    final date = DateTime.tryParse(value?.toString().split('T').first ?? '');
    if (date == null) return 'Unknown month';
    const months = [
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec',
    ];
    return '${months[date.month - 1]} ${date.year}';
  }

  String get _currentMonth {
    final now = DateTime.now();
    return '${now.year.toString().padLeft(4, '0')}-${now.month.toString().padLeft(2, '0')}-01';
  }

  Future<void> _load() async {
    if (mounted)
      setState(() {
        _loading = true;
        _error = null;
      });
    try {
      final data = await Api().getWardenStudentRentRecords(widget.studentId);
      if (!mounted) return;
      setState(() {
        _records = List<dynamic>.from(data['records'] as List? ?? const [])
            .whereType<Map>()
            .map((row) => Map<String, dynamic>.from(row))
            .toList();
        _loading = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = error.toString().replaceFirst('Exception: ', '');
      });
    }
  }

  Future<void> _setUpCurrentMonth() async {
    if (widget.bookingId == null) {
      _mobileToast(
        context,
        'Assign a room before creating this rent record.',
        error: true,
      );
      return;
    }
    try {
      await Api().setupWardenRentRecord(
        bookingId: widget.bookingId!,
        billingMonth: _currentMonth,
      );
      await _load();
      if (mounted) _mobileToast(context, 'Current-month rent record created.');
    } catch (error) {
      if (mounted)
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
    }
  }

  Future<void> _recordCash(Map<String, dynamic> record) async {
    final invoiceId = int.tryParse('${record['invoice_id']}');
    if (invoiceId == null) return;
    final existingId = int.tryParse('${record['latest_cash_payment_id']}');
    final amountController = TextEditingController(
      text: _amount(
        existingId == null
            ? record['remaining_amount']
            : record['latest_cash_amount'],
      ).toStringAsFixed(0),
    );
    final noteController = TextEditingController(
      text: record['latest_cash_note']?.toString() ?? '',
    );
    var receivedOn =
        DateTime.tryParse(
          record['latest_cash_received_on']?.toString().split('T').first ?? '',
        ) ??
        DateTime.now();
    final saved = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => StatefulBuilder(
        builder: (_, setDialogState) => AlertDialog(
          backgroundColor: const Color(0xFF07393C),
          title: Text(
            existingId == null ? 'Mark cash rent paid' : 'Edit cash payment',
            style: GoogleFonts.dmSans(
              color: Colors.white,
              fontWeight: FontWeight.w800,
            ),
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '${_monthLabel(record['billing_month'])} · ${_money(record['remaining_amount'])} remaining',
                style: GoogleFonts.dmSans(color: Colors.white70, fontSize: 10),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: amountController,
                keyboardType: const TextInputType.numberWithOptions(
                  decimal: true,
                ),
                style: const TextStyle(color: Colors.white),
                decoration: _inputDecoration(
                  'Cash amount (PKR)',
                  icon: Icons.payments_outlined,
                ),
              ),
              const SizedBox(height: 10),
              OutlinedButton.icon(
                onPressed: () async {
                  final picked = await showDatePicker(
                    context: dialogContext,
                    initialDate: receivedOn,
                    firstDate: DateTime(2020),
                    lastDate: DateTime(DateTime.now().year + 1),
                  );
                  if (picked != null) setDialogState(() => receivedOn = picked);
                },
                icon: const Icon(Icons.event_outlined, size: 17),
                label: Text('Received on: ${_isoDate(receivedOn)}'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: _mint,
                  side: BorderSide(color: _mint.withOpacity(0.55)),
                ),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: noteController,
                maxLines: 2,
                style: const TextStyle(color: Colors.white),
                decoration: _inputDecoration(
                  'Note / receipt book reference',
                  icon: Icons.note_outlined,
                ),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(dialogContext),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () async {
                final amount = double.tryParse(amountController.text.trim());
                if (amount == null || amount <= 0) {
                  _mobileToast(
                    context,
                    'Enter a valid cash amount.',
                    error: true,
                  );
                  return;
                }
                try {
                  if (existingId == null) {
                    await Api().recordWardenCashFeePayment(
                      invoiceId: invoiceId,
                      amount: amount,
                      receivedOn: _isoDate(receivedOn),
                      note: noteController.text,
                    );
                  } else {
                    await Api().updateWardenCashFeePayment(
                      paymentId: existingId,
                      amount: amount,
                      receivedOn: _isoDate(receivedOn),
                      note: noteController.text,
                    );
                  }
                  if (dialogContext.mounted) Navigator.pop(dialogContext, true);
                } catch (error) {
                  if (mounted)
                    _mobileToast(
                      context,
                      error.toString().replaceFirst('Exception: ', ''),
                      error: true,
                    );
                }
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: _teal,
                foregroundColor: Colors.white,
              ),
              child: Text(existingId == null ? 'Mark paid' : 'Save changes'),
            ),
          ],
        ),
      ),
    );
    amountController.dispose();
    noteController.dispose();
    if (saved == true) {
      await _load();
      if (mounted) _mobileToast(context, 'Cash payment saved.');
    }
  }

  Future<void> _viewReceipt(Map<String, dynamic> record) async {
    final receiptId = int.tryParse('${record['receipt_id']}');
    if (receiptId == null) return;
    try {
      final response = await Api().downloadFeeReceipt(receiptId);
      if (!mounted) return;
      final isImage = (response.headers['content-type'] ?? '').startsWith(
        'image/',
      );
      await showDialog<void>(
        context: context,
        builder: (_) => Dialog(
          child: isImage
              ? InteractiveViewer(child: Image.memory(response.bodyBytes))
              : SizedBox(
                  width: 700,
                  height: 700,
                  child: PdfPreview(build: (_) async => response.bodyBytes),
                ),
        ),
      );
    } catch (error) {
      if (mounted)
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
    }
  }

  Future<void> _verifyReceipt(Map<String, dynamic> record) async {
    final receiptId = int.tryParse('${record['receipt_id']}');
    if (receiptId == null) return;
    try {
      await Api().reviewWardenFeeReceipt(
        receiptId: receiptId,
        status: 'verified',
        note: 'Verified by warden after receipt review.',
      );
      await _load();
      if (mounted) _mobileToast(context, 'Receipt verified.');
    } catch (error) {
      if (mounted)
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
    }
  }

  @override
  Widget build(BuildContext context) {
    final currentMonthExists = _records.any(
      (record) =>
          record['billing_month']?.toString().split('T').first == _currentMonth,
    );
    return _WardenPage(
      title: 'Student Rent Details',
      hostelName: widget.hostelName,
      child: RefreshIndicator(
        onRefresh: _load,
        child: ListView(
          physics: const AlwaysScrollableScrollPhysics(
            parent: BouncingScrollPhysics(),
          ),
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
          children: [
            _GlassCard(
              child: Row(
                children: [
                  CircleAvatar(
                    radius: 25,
                    backgroundColor: _teal.withOpacity(0.2),
                    child: Text(
                      widget.studentName.isEmpty
                          ? 'S'
                          : widget.studentName[0].toUpperCase(),
                      style: GoogleFonts.dmSans(
                        color: _mint,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          widget.studentName,
                          style: GoogleFonts.dmSans(
                            color: Colors.white,
                            fontWeight: FontWeight.w800,
                            fontSize: 15,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          _allocationDetail(
                            floor: widget.floor,
                            room: widget.room,
                            bed: widget.bed,
                          ),
                          style: GoogleFonts.dmSans(
                            color: Colors.white60,
                            fontSize: 9,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            if (!currentMonthExists) ...[
              OutlinedButton.icon(
                onPressed: _setUpCurrentMonth,
                icon: const Icon(Icons.add_card_outlined),
                label: const Text('Set up current-month rent'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: _mint,
                  side: BorderSide(color: _mint.withOpacity(0.55)),
                ),
              ),
              const SizedBox(height: 12),
            ],
            Text(
              'All months & years',
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontWeight: FontWeight.w800,
                fontSize: 13,
              ),
            ),
            const SizedBox(height: 9),
            if (_loading)
              const Padding(
                padding: EdgeInsets.all(32),
                child: Center(child: CircularProgressIndicator()),
              )
            else if (_error != null)
              Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  children: [
                    Text(
                      'Could not load rent history.',
                      style: GoogleFonts.dmSans(color: Colors.white70),
                    ),
                    TextButton(onPressed: _load, child: const Text('Retry')),
                  ],
                ),
              )
            else if (_records.isEmpty)
              Padding(
                padding: const EdgeInsets.all(24),
                child: Text(
                  'No rent record yet.',
                  textAlign: TextAlign.center,
                  style: GoogleFonts.dmSans(color: Colors.white60),
                ),
              )
            else
              ..._records.map(_historyCard),
          ],
        ),
      ),
    );
  }

  Widget _historyCard(Map<String, dynamic> record) {
    final remaining = _amount(record['remaining_amount']);
    final paid = remaining <= 0;
    final receiptSubmitted = record['receipt_status'] == 'submitted';
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: _GlassCard(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    _monthLabel(record['billing_month']),
                    style: GoogleFonts.dmSans(
                      color: Colors.white,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ),
                _historyStatus(paid ? 'Paid' : 'Due', paid ? _mint : _orange),
              ],
            ),
            const SizedBox(height: 10),
            Row(
              children: [
                _historyAmount('Rent', _money(record['invoice_amount'])),
                _historyAmount('Paid', _money(record['paid_amount'])),
                _historyAmount('Balance', _money(remaining)),
              ],
            ),
            if (record['latest_cash_payment_id'] != null) ...[
              const SizedBox(height: 9),
              Text(
                'Cash received: ${record['latest_cash_received_on']?.toString().split('T').first ?? '—'}',
                style: GoogleFonts.dmSans(color: Colors.white60, fontSize: 8.5),
              ),
            ],
            if (record['receipt_id'] != null) ...[
              const SizedBox(height: 8),
              Row(
                children: [
                  Icon(Icons.receipt_long_outlined, color: _mint, size: 16),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      'Student receipt · ${record['receipt_status']}',
                      style: GoogleFonts.dmSans(
                        color: Colors.white70,
                        fontSize: 8.5,
                      ),
                    ),
                  ),
                  TextButton(
                    onPressed: () => _viewReceipt(record),
                    child: const Text('View'),
                  ),
                ],
              ),
            ],
            const SizedBox(height: 10),
            Row(
              children: [
                if (receiptSubmitted) ...[
                  Expanded(
                    child: OutlinedButton(
                      onPressed: () => _verifyReceipt(record),
                      style: OutlinedButton.styleFrom(foregroundColor: _mint),
                      child: const Text('Verify receipt'),
                    ),
                  ),
                  const SizedBox(width: 8),
                ],
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: paid && record['latest_cash_payment_id'] == null
                        ? null
                        : () => _recordCash(record),
                    icon: Icon(
                      record['latest_cash_payment_id'] == null
                          ? Icons.payments_outlined
                          : Icons.edit_outlined,
                      size: 16,
                    ),
                    label: Text(
                      record['latest_cash_payment_id'] == null
                          ? 'Mark as Paid'
                          : 'Edit cash',
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _teal,
                      foregroundColor: Colors.white,
                      disabledBackgroundColor: _teal.withOpacity(0.25),
                      elevation: 0,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _historyAmount(String label, String value) => Expanded(
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: GoogleFonts.dmSans(color: Colors.white60, fontSize: 8),
        ),
        const SizedBox(height: 2),
        Text(
          value,
          style: GoogleFonts.dmSans(
            color: Colors.white,
            fontWeight: FontWeight.w700,
            fontSize: 10,
          ),
        ),
      ],
    ),
  );

  Widget _historyStatus(String text, Color color) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
    decoration: BoxDecoration(
      color: color.withOpacity(0.14),
      borderRadius: BorderRadius.circular(20),
    ),
    child: Text(
      text,
      style: GoogleFonts.dmSans(
        color: color,
        fontSize: 8,
        fontWeight: FontWeight.w800,
      ),
    ),
  );
}

class WardenEmptyBedsScreen extends StatefulWidget {
  final String hostelName;

  const WardenEmptyBedsScreen({super.key, required this.hostelName});

  @override
  State<WardenEmptyBedsScreen> createState() => _WardenEmptyBedsScreenState();
}

class _WardenEmptyBedsScreenState extends State<WardenEmptyBedsScreen> {
  final List<Map<String, String>> _beds = [
    {'room': '104', 'bed': '1'},
    {'room': '104', 'bed': '2'},
    {'room': '206', 'bed': '1'},
  ];

  Future<void> _addEmptyBed() async {
    final result = await Navigator.push<Map<String, String>>(
      context,
      MaterialPageRoute(
        builder: (_) => _AddEmptyBedScreen(hostelName: widget.hostelName),
      ),
    );

    if (!mounted || result == null) {
      return;
    }

    setState(() {
      _beds.add(result);
    });

    _mobileToast(context, 'Empty bed added.');
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: 'Empty Beds',
      hostelName: widget.hostelName,
      floatingActionButton: FloatingActionButton.extended(
        backgroundColor: _teal,
        foregroundColor: Colors.white,
        onPressed: _addEmptyBed,
        icon: const Icon(Icons.add_rounded),
        label: const Text('Add Empty Bed'),
      ),
      child: ListView.separated(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
        itemCount: _beds.length,
        separatorBuilder: (_, __) => const SizedBox(height: 9),
        itemBuilder: (_, index) {
          final bed = _beds[index];

          return _GlassCard(
            child: Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: _teal.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(13),
                  ),
                  child: Icon(Icons.bed_rounded, color: _mint),
                ),

                const SizedBox(width: 11),

                Expanded(
                  child: Text(
                    'Room ${bed['room']} · Bed ${bed['bed']}',
                    style: GoogleFonts.dmSans(
                      color: Colors.white,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ),

                TextButton(
                  onPressed: () {
                    setState(() {
                      _beds.removeAt(index);
                    });

                    _mobileToast(context, 'Bed marked occupied.');
                  },
                  child: Text(
                    'Mark\nOccupied',
                    textAlign: TextAlign.center,
                    style: GoogleFonts.dmSans(
                      color: _mint,
                      fontSize: 8,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}

class _AddEmptyBedScreen extends StatefulWidget {
  final String hostelName;

  const _AddEmptyBedScreen({required this.hostelName});

  @override
  State<_AddEmptyBedScreen> createState() => _AddEmptyBedScreenState();
}

class _AddEmptyBedScreenState extends State<_AddEmptyBedScreen> {
  final _roomCtrl = TextEditingController();

  final _bedCtrl = TextEditingController();

  @override
  void dispose() {
    _roomCtrl.dispose();
    _bedCtrl.dispose();
    super.dispose();
  }

  void _save() {
    if (_roomCtrl.text.trim().isEmpty || _bedCtrl.text.trim().isEmpty) {
      _mobileToast(context, 'Enter room and bed number.', error: true);
      return;
    }

    Navigator.pop(context, <String, String>{
      'room': _roomCtrl.text.trim(),
      'bed': _bedCtrl.text.trim(),
    });
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: 'Add Empty Bed',
      hostelName: widget.hostelName,
      child: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
        child: _GlassCard(
          child: Column(
            children: [
              TextField(
                controller: _roomCtrl,
                style: const TextStyle(color: Colors.white),
                decoration: _inputDecoration(
                  'Room number',
                  icon: Icons.meeting_room_outlined,
                ),
              ),

              const SizedBox(height: 10),

              TextField(
                controller: _bedCtrl,
                style: const TextStyle(color: Colors.white),
                decoration: _inputDecoration(
                  'Bed number',
                  icon: Icons.bed_outlined,
                ),
              ),

              const SizedBox(height: 16),

              _PrimaryButton(
                text: 'Add Available Bed',
                icon: Icons.add_rounded,
                onPressed: _save,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// 8. CHAT
// =============================================================================

class WardenChatScreen extends StatefulWidget {
  final String hostelName;

  const WardenChatScreen({super.key, required this.hostelName});

  @override
  State<WardenChatScreen> createState() => _WardenChatScreenState();
}

class _WardenChatScreenState extends State<WardenChatScreen> {
  List<Map<String, dynamic>> _chats = [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadStudents();
  }

  Future<void> _loadStudents() async {
    try {
      final rows = await Api().getWardenStudents();
      if (!mounted) return;
      setState(() {
        _chats = rows.whereType<Map>().map((row) {
          final student = Map<String, dynamic>.from(row);
          return {
            'studentId': student['id'],
            'name': student['name'] ?? 'Student',
            'type': 'Student',
            'floor': student['floor_label'],
            'room': student['room_number'],
            'bed': student['bed_number'],
          };
        }).toList();
        _loading = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = error.toString().replaceFirst('Exception: ', '');
      });
    }
  }

  Future<void> _openChat(Map<String, dynamic> chat) async {
    final studentId = int.tryParse(chat['studentId'].toString());
    if (studentId == null) return;
    try {
      final result = await Api().openWardenConversation(studentId);
      final conversation = Map<String, dynamic>.from(
        result['conversation'] as Map,
      );
      final conversationId = int.tryParse(conversation['id'].toString());
      if (conversationId == null || !mounted) return;
      await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => _ChatDetailScreen(
            hostelName: widget.hostelName,
            name: chat['name'].toString(),
            type: chat['type'].toString(),
            conversationId: conversationId,
          ),
        ),
      );
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(error.toString().replaceFirst('Exception: ', '')),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: 'Chat',
      hostelName: widget.hostelName,
      child: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
          ? Center(
              child: TextButton(
                onPressed: _loadStudents,
                child: const Text('Retry'),
              ),
            )
          : ListView.separated(
              physics: const BouncingScrollPhysics(),
              padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
              itemCount: _chats.length,
              separatorBuilder: (_, __) => const SizedBox(height: 9),
              itemBuilder: (_, index) {
                final chat = _chats[index];

                final name = chat['name'].toString();

                return InkWell(
                  onTap: () {
                    _openChat(chat);
                  },
                  borderRadius: BorderRadius.circular(20),
                  child: _GlassCard(
                    child: Row(
                      children: [
                        CircleAvatar(
                          radius: 22,
                          backgroundColor: _teal.withOpacity(0.20),
                          child: Text(
                            name.isEmpty ? '?' : name[0].toUpperCase(),
                            style: GoogleFonts.dmSans(
                              color: _mint,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ),

                        const SizedBox(width: 11),

                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                name,
                                style: GoogleFonts.dmSans(
                                  color: Colors.white,
                                  fontWeight: FontWeight.w800,
                                ),
                              ),

                              const SizedBox(height: 3),

                              Text(
                                '${_allocationDetail(floor: chat['floor'], room: chat['room'], bed: chat['bed'])} · Start conversation',
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: GoogleFonts.dmSans(
                                  color: Colors.white.withOpacity(0.5),
                                  fontSize: 8.5,
                                ),
                              ),
                            ],
                          ),
                        ),

                        Icon(Icons.chevron_right_rounded, color: _mint),
                      ],
                    ),
                  ),
                );
              },
            ),
    );
  }
}

class _ChatDetailScreen extends StatefulWidget {
  final String hostelName;
  final String name;
  final String type;
  final int conversationId;

  const _ChatDetailScreen({
    required this.hostelName,
    required this.name,
    required this.type,
    required this.conversationId,
  });

  @override
  State<_ChatDetailScreen> createState() => _ChatDetailScreenState();
}

class _ChatDetailScreenState extends State<_ChatDetailScreen> {
  final _messageCtrl = TextEditingController();

  final _scrollController = ScrollController();
  List<Map<String, dynamic>> _messages = [];
  bool _loading = true;
  bool _sending = false;

  @override
  void initState() {
    super.initState();
    _loadMessages();
  }

  Future<void> _loadMessages() async {
    try {
      final rows = await Api().getWardenMessages(widget.conversationId);
      if (!mounted) return;
      setState(() {
        _messages = rows.whereType<Map>().map((row) {
          final message = Map<String, dynamic>.from(row);
          return {
            'me': message['is_mine'] == true,
            'text': message['body'] ?? '',
          };
        }).toList();
        _loading = false;
      });
      _scrollToBottom();
    } catch (_) {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  void dispose() {
    _messageCtrl.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final text = _messageCtrl.text.trim();

    if (text.isEmpty || _sending) {
      return;
    }
    setState(() => _sending = true);
    try {
      final response = await Api().sendWardenMessage(
        conversationId: widget.conversationId,
        message: text,
      );
      final rawMessage = Map<String, dynamic>.from(response['message'] as Map);
      if (!mounted) return;
      setState(() {
        _messages.add({
          'me': rawMessage['is_mine'] == true,
          'text': rawMessage['body'] ?? text,
        });
        _sending = false;
      });
      _messageCtrl.clear();
      _scrollToBottom();
    } catch (error) {
      if (!mounted) return;
      setState(() => _sending = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(error.toString().replaceFirst('Exception: ', '')),
        ),
      );
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) {
        return;
      }

      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOut,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: widget.name,
      hostelName: '${widget.type} · ${widget.hostelName}',
      child: Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              physics: const BouncingScrollPhysics(),
              padding: const EdgeInsets.fromLTRB(16, 6, 16, 12),
              itemCount: _loading ? 0 : _messages.length,
              itemBuilder: (_, index) {
                final message = _messages[index];

                final me = message['me'] == true;

                return Align(
                  alignment: me ? Alignment.centerRight : Alignment.centerLeft,
                  child: Container(
                    constraints: const BoxConstraints(maxWidth: 250),
                    margin: const EdgeInsets.only(bottom: 8),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 10,
                    ),
                    decoration: BoxDecoration(
                      color: me ? _teal : _glass,
                      borderRadius: BorderRadius.circular(16),
                      border: me
                          ? null
                          : Border.all(color: Colors.white.withOpacity(0.10)),
                    ),
                    child: Text(
                      message['text'].toString(),
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontSize: 10.5,
                        height: 1.4,
                      ),
                    ),
                  ),
                );
              },
            ),
          ),

          SafeArea(
            top: false,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _messageCtrl,
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) {
                        _send();
                      },
                      style: const TextStyle(color: Colors.white),
                      decoration: _inputDecoration(
                        'Type a message...',
                        icon: Icons.chat_bubble_outline,
                      ),
                    ),
                  ),

                  const SizedBox(width: 8),

                  InkWell(
                    onTap: _sending ? null : _send,
                    borderRadius: BorderRadius.circular(50),
                    child: Container(
                      width: 46,
                      height: 46,
                      decoration: BoxDecoration(
                        color: _teal,
                        shape: BoxShape.circle,
                      ),
                      child: _sending
                          ? const Padding(
                              padding: EdgeInsets.all(14),
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: Colors.white,
                              ),
                            )
                          : const Icon(
                              Icons.send_rounded,
                              color: Colors.white,
                              size: 19,
                            ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// =============================================================================
// 9. CLEANING SCHEDULE
// =============================================================================

class WardenCleaningScheduleScreen extends StatefulWidget {
  final String hostelName;

  const WardenCleaningScheduleScreen({super.key, required this.hostelName});

  @override
  State<WardenCleaningScheduleScreen> createState() =>
      _WardenCleaningScheduleScreenState();
}

class _WardenCleaningScheduleScreenState
    extends State<WardenCleaningScheduleScreen> {
  List<Map<String, dynamic>> _schedules = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final rows = await Api().getCleaningSchedule();
      if (!mounted) return;
      setState(() {
        _schedules = rows.whereType<Map>().map((row) {
          final item = Map<String, dynamic>.from(row);
          final date = DateTime.tryParse(
            item['scheduled_date']?.toString() ?? '',
          );
          return {
            ...item,
            'day': date == null
                ? 'Scheduled cleaning'
                : [
                    'Monday',
                    'Tuesday',
                    'Wednesday',
                    'Thursday',
                    'Friday',
                    'Saturday',
                    'Sunday',
                  ][date.weekday - 1],
            'completed': item['status'] == 'completed',
          };
        }).toList();
        _loading = false;
      });
    } catch (error) {
      if (mounted) {
        setState(() => _loading = false);
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
      }
    }
  }

  Future<void> _addSchedule() async {
    final result = await Navigator.push<Map<String, dynamic>>(
      context,
      MaterialPageRoute(
        builder: (_) =>
            _AddCleaningScheduleScreen(hostelName: widget.hostelName),
      ),
    );

    if (!mounted || result == null) {
      return;
    }

    _mobileToast(context, 'Cleaning schedule added.');
    await _load();
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: 'Cleaning Schedule',
      hostelName: widget.hostelName,
      floatingActionButton: FloatingActionButton.extended(
        backgroundColor: _teal,
        foregroundColor: Colors.white,
        onPressed: _addSchedule,
        icon: const Icon(Icons.add_rounded),
        label: const Text('Add Schedule'),
      ),
      child: ListView.separated(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
        itemCount: _loading
            ? 1
            : _schedules.isEmpty
            ? 1
            : _schedules.length,
        separatorBuilder: (_, __) => const SizedBox(height: 9),
        itemBuilder: (_, index) {
          if (_loading)
            return const Padding(
              padding: EdgeInsets.all(28),
              child: Center(child: CircularProgressIndicator()),
            );
          if (_schedules.isEmpty)
            return _emptyMessage('No cleaning schedule has been added.');
          final schedule = _schedules[index];

          final completed = schedule['completed'] == true;

          return _GlassCard(
            child: Column(
              children: [
                Row(
                  children: [
                    Container(
                      width: 44,
                      height: 44,
                      decoration: BoxDecoration(
                        color: _teal.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(13),
                      ),
                      child: Icon(
                        Icons.cleaning_services_rounded,
                        color: _mint,
                      ),
                    ),

                    const SizedBox(width: 11),

                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            schedule['day'].toString(),
                            style: GoogleFonts.dmSans(
                              color: Colors.white,
                              fontWeight: FontWeight.w800,
                            ),
                          ),

                          Text(
                            schedule['area'].toString(),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: GoogleFonts.dmSans(
                              color: Colors.white.withOpacity(0.5),
                              fontSize: 8.5,
                            ),
                          ),

                          Text(
                            schedule['shift'].toString(),
                            style: GoogleFonts.dmSans(
                              color: _mint,
                              fontSize: 8,
                            ),
                          ),
                        ],
                      ),
                    ),

                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 9,
                        vertical: 5,
                      ),
                      decoration: BoxDecoration(
                        color: completed
                            ? _green.withOpacity(0.15)
                            : _orange.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        completed ? 'Completed' : 'Pending',
                        style: GoogleFonts.dmSans(
                          color: completed ? _green : _orange,
                          fontSize: 8,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                    ),
                  ],
                ),

                if (!completed) ...[
                  const SizedBox(height: 10),

                  SizedBox(
                    width: double.infinity,
                    child: OutlinedButton.icon(
                      onPressed: () async {
                        try {
                          await Api().updateCleaningSchedule(
                            scheduleId: int.parse(schedule['id'].toString()),
                            status: 'completed',
                          );
                          if (!mounted) return;
                          _mobileToast(context, 'Cleaning marked complete.');
                          await _load();
                        } catch (error) {
                          if (mounted)
                            _mobileToast(
                              context,
                              error.toString().replaceFirst('Exception: ', ''),
                              error: true,
                            );
                        }
                      },
                      icon: const Icon(
                        Icons.check_circle_outline_rounded,
                        size: 17,
                      ),
                      label: const Text('Mark Complete'),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: _mint,
                        side: BorderSide(color: _mint.withOpacity(0.35)),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(13),
                        ),
                      ),
                    ),
                  ),
                ],
              ],
            ),
          );
        },
      ),
    );
  }
}

class _AddCleaningScheduleScreen extends StatefulWidget {
  final String hostelName;

  const _AddCleaningScheduleScreen({required this.hostelName});

  @override
  State<_AddCleaningScheduleScreen> createState() =>
      _AddCleaningScheduleScreenState();
}

class _AddCleaningScheduleScreenState
    extends State<_AddCleaningScheduleScreen> {
  String? _day;
  String _locationType = 'Room';
  String? _floor;
  String? _room;
  String? _area;
  TimeOfDay? _time;
  String _shift = '';
  List<Map<String, dynamic>> _rooms = [];
  bool _loadingRooms = true;

  @override
  void initState() {
    super.initState();
    _loadRooms();
  }

  Future<void> _loadRooms() async {
    try {
      final rows = await Api().getWardenRooms();
      if (!mounted) return;
      setState(() {
        _rooms = rows
            .whereType<Map>()
            .map((row) => Map<String, dynamic>.from(row))
            .toList();
        _loadingRooms = false;
      });
    } catch (error) {
      if (mounted) {
        setState(() => _loadingRooms = false);
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
      }
    }
  }

  List<String> get _floors =>
      _rooms
          .map((room) => room['floor_label']?.toString() ?? '')
          .where((floor) => floor.isNotEmpty)
          .toSet()
          .toList()
        ..sort();

  List<String> get _roomsForFloor =>
      _rooms
          .where(
            (room) =>
                _floor == null || room['floor_label']?.toString() == _floor,
          )
          .map((room) => room['room_number']?.toString() ?? '')
          .where((room) => room.isNotEmpty)
          .toSet()
          .toList()
        ..sort();

  @override
  void dispose() {
    super.dispose();
  }

  Future<void> _save() async {
    final location = _locationType == 'Room'
        ? (_floor == null || _room == null ? '' : '$_floor · Room $_room')
        : (_area ?? '');
    if (_day == null || location.isEmpty || _time == null) {
      _mobileToast(
        context,
        'Select day, location and cleaning time.',
        error: true,
      );
      return;
    }

    final weekdays = const {
      'Monday': DateTime.monday,
      'Tuesday': DateTime.tuesday,
      'Wednesday': DateTime.wednesday,
      'Thursday': DateTime.thursday,
      'Friday': DateTime.friday,
      'Saturday': DateTime.saturday,
      'Sunday': DateTime.sunday,
    };
    final today = DateTime.now();
    final selectedWeekday = weekdays[_day]!;
    final offset = (selectedWeekday - today.weekday + 7) % 7;
    final date = DateTime(
      today.year,
      today.month,
      today.day,
    ).add(Duration(days: offset));
    final dateKey =
        '${date.year.toString().padLeft(4, '0')}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')}';
    try {
      await Api().createCleaningSchedule(
        area: location,
        scheduledDate: dateKey,
        shift: _time!.format(context),
      );
      if (mounted) Navigator.pop(context, <String, dynamic>{'saved': true});
    } catch (error) {
      if (mounted)
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
    }
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: 'Add Cleaning Schedule',
      hostelName: widget.hostelName,
      child: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
        child: _GlassCard(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (_loadingRooms)
                const Padding(
                  padding: EdgeInsets.only(bottom: 12),
                  child: LinearProgressIndicator(),
                ),
              _scheduleDropdown(
                value: _day,
                hint: 'Choose day',
                icon: Icons.calendar_today_outlined,
                items: const [
                  'Monday',
                  'Tuesday',
                  'Wednesday',
                  'Thursday',
                  'Friday',
                  'Saturday',
                  'Sunday',
                ].map(_scheduleItem).toList(),
                onChanged: (value) => setState(() => _day = value),
              ),

              const SizedBox(height: 10),

              _scheduleDropdown(
                value: _locationType,
                hint: 'Clean rooms or area?',
                icon: Icons.cleaning_services_outlined,
                items: const ['Room', 'Area'].map(_scheduleItem).toList(),
                onChanged: (value) => setState(() => _locationType = value!),
              ),

              const SizedBox(height: 10),
              _scheduleDropdown(
                value: _floor,
                hint: 'Floor number',
                icon: Icons.layers_outlined,
                items: _floors
                    .map((floor) => _scheduleItem(floor, floor))
                    .toList(),
                onChanged: (value) => setState(() {
                  _floor = value;
                  _room = null;
                }),
              ),
              const SizedBox(height: 10),
              if (_locationType == 'Room') ...[
                const SizedBox(height: 10),
                _scheduleDropdown(
                  value: _room,
                  hint: 'Room number',
                  icon: Icons.meeting_room_outlined,
                  items: _roomsForFloor
                      .map((room) => _scheduleItem(room, 'Room $room'))
                      .toList(),
                  onChanged: (value) => setState(() => _room = value),
                ),
              ] else
                _scheduleDropdown(
                  value: _area,
                  hint: 'Area',
                  icon: Icons.cleaning_services_outlined,
                  items: const [
                    'Corridor',
                    'Bathroom',
                    'Study Room',
                    'Prayer Room',
                    'Kitchen',
                  ].map(_scheduleItem).toList(),
                  onChanged: (value) => setState(() => _area = value),
                ),

              const SizedBox(height: 14),

              OutlinedButton.icon(
                onPressed: () async {
                  final value = await showTimePicker(
                    context: context,
                    useRootNavigator: false,
                    initialEntryMode: TimePickerEntryMode.dial,
                    orientation: Orientation.portrait,
                    initialTime: _time ?? TimeOfDay.now(),
                  );
                  if (value != null) setState(() => _time = value);
                },
                icon: const Icon(Icons.access_time_rounded),
                label: Text(
                  _time == null
                      ? 'Select time (AM/PM)'
                      : _time!.format(context),
                ),
                style: OutlinedButton.styleFrom(
                  foregroundColor: Colors.white,
                  minimumSize: const Size.fromHeight(48),
                ),
              ),

              const SizedBox(height: 18),

              _PrimaryButton(
                text: 'Create Schedule',
                icon: Icons.cleaning_services_rounded,
                onPressed: _save,
              ),
            ],
          ),
        ),
      ),
    );
  }

  DropdownMenuItem<String> _scheduleItem(String value, [String? label]) {
    return DropdownMenuItem(
      value: value,
      child: Text(
        label ?? value,
        style: GoogleFonts.dmSans(color: Colors.white),
      ),
    );
  }

  Widget _scheduleDropdown({
    required String? value,
    required String hint,
    required IconData icon,
    required List<DropdownMenuItem<String>> items,
    required ValueChanged<String?> onChanged,
  }) {
    return DropdownButtonFormField<String>(
      value: value,
      isExpanded: true,
      dropdownColor: const Color(0xFF07393C),
      iconEnabledColor: Colors.white70,
      style: GoogleFonts.dmSans(color: Colors.white),
      hint: Text(hint, style: GoogleFonts.dmSans(color: Colors.white70)),
      decoration: _inputDecoration(hint, icon: icon),
      items: items,
      onChanged: onChanged,
    );
  }

  Widget _shiftButton(String value) {
    final selected = _shift == value;

    return Expanded(
      child: InkWell(
        onTap: () {
          setState(() {
            _shift = value;
          });
        },
        borderRadius: BorderRadius.circular(12),
        child: Container(
          height: 40,
          decoration: BoxDecoration(
            color: selected ? _teal : Colors.black.withOpacity(0.25),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: selected
                  ? _mint.withOpacity(0.35)
                  : Colors.white.withOpacity(0.09),
            ),
          ),
          child: Center(
            child: Text(
              value,
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontSize: 8.5,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// =============================================================================
// ROOM / BED CHANGE REQUESTS
// =============================================================================

class WardenRoomChangeRequestsScreen extends StatefulWidget {
  final String hostelName;

  const WardenRoomChangeRequestsScreen({
    super.key,
    required this.hostelName,
  });

  @override
  State<WardenRoomChangeRequestsScreen> createState() =>
      _WardenRoomChangeRequestsScreenState();
}

class _WardenRoomChangeRequestsScreenState
    extends State<WardenRoomChangeRequestsScreen> {
  bool _loading = true;
  int? _decidingRequestId;
  List<Map<String, dynamic>> _requests = [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final records = await Api().getWardenRoomChangeRequests();
      if (!mounted) return;
      setState(() {
        _requests = records
            .whereType<Map>()
            .map((record) => Map<String, dynamic>.from(record))
            .toList();
        _loading = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _loading = false);
      _mobileToast(
        context,
        error.toString().replaceFirst('Exception: ', ''),
        error: true,
      );
    }
  }

  String _date(dynamic value) {
    final parsed = DateTime.tryParse(value?.toString() ?? '');
    if (parsed == null) return '—';
    const months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    ];
    final local = parsed.toLocal();
    return '${local.day.toString().padLeft(2, '0')} ${months[local.month - 1]} ${local.year}';
  }

  Future<void> _decide(Map<String, dynamic> request, String decision) async {
    final requestId = int.tryParse(request['id'].toString());
    if (requestId == null || _decidingRequestId != null) return;
    setState(() => _decidingRequestId = requestId);
    try {
      await Api().decideWardenRoomChangeRequest(
        requestId: requestId,
        status: decision,
      );
      if (!mounted) return;
      _mobileToast(
        context,
        decision == 'approved'
            ? 'Approved. The student allocation has been updated.'
            : 'Request rejected.',
      );
      await _load();
    } catch (error) {
      if (mounted) {
        _mobileToast(
          context,
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
      }
    } finally {
      if (mounted) setState(() => _decidingRequestId = null);
    }
  }

  Widget _requestCard(Map<String, dynamic> request) {
    final requestId = int.tryParse(request['id'].toString());
    final status = (request['status'] ?? 'pending').toString().toLowerCase();
    final isPending = status == 'pending';
    final isBusy = _decidingRequestId == requestId;
    final target = 'Floor ${request['floor_label'] ?? '—'} · '
        'Room ${request['room_number'] ?? '—'} · '
        'Bed ${request['requested_bed_number'] ?? '—'}';
    final statusColor = status == 'approved'
        ? _green
        : status == 'rejected'
        ? _red
        : _orange;

    return _GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              CircleAvatar(
                radius: 19,
                backgroundColor: _teal.withOpacity(0.30),
                child: Text(
                  (request['student_name']?.toString().trim().isNotEmpty ?? false)
                      ? request['student_name'].toString().trim()[0].toUpperCase()
                      : 'S',
                  style: GoogleFonts.dmSans(
                    color: _mint,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      request['student_name']?.toString() ?? 'Student',
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      'Submitted ${_date(request['created_at'])}',
                      style: GoogleFonts.dmSans(
                        color: Colors.white70,
                        fontSize: 9.5,
                      ),
                    ),
                  ],
                ),
              ),
              Text(
                status.toUpperCase(),
                style: GoogleFonts.dmSans(
                  color: statusColor,
                  fontSize: 9,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            'Requested move',
            style: GoogleFonts.dmSans(
              color: Colors.white.withOpacity(0.55),
              fontSize: 9,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 3),
          Text(
            target,
            style: GoogleFonts.dmSans(
              color: _mint,
              fontWeight: FontWeight.w700,
              fontSize: 11,
            ),
          ),
          const SizedBox(height: 10),
          Text(
            request['reason']?.toString() ?? 'No reason provided.',
            style: GoogleFonts.dmSans(color: Colors.white70, fontSize: 10),
          ),
          if (isPending) ...[
            const SizedBox(height: 14),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: isBusy ? null : () => _decide(request, 'rejected'),
                    icon: const Icon(Icons.close_rounded, size: 16),
                    label: const Text('Reject'),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: _red,
                      side: const BorderSide(color: _red),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: isBusy ? null : () => _decide(request, 'approved'),
                    icon: isBusy
                        ? const SizedBox(
                            width: 14,
                            height: 14,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : const Icon(Icons.check_rounded, size: 16),
                    label: const Text('Approve'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _teal,
                      foregroundColor: Colors.white,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return _WardenPage(
      title: 'Room change requests',
      hostelName: widget.hostelName,
      child: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _load,
              child: _requests.isEmpty
                  ? ListView(
                      padding: const EdgeInsets.all(24),
                      children: [_emptyMessage('No room or bed change requests yet.')],
                    )
                  : ListView.separated(
                      physics: const AlwaysScrollableScrollPhysics(),
                      padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
                      itemCount: _requests.length,
                      separatorBuilder: (_, __) => const SizedBox(height: 10),
                      itemBuilder: (_, index) => _requestCard(_requests[index]),
                    ),
            ),
    );
  }
}

// =============================================================================
// EMPTY STATE
// =============================================================================

Widget _emptyMessage(String text) {
  return Padding(
    padding: const EdgeInsets.symmetric(vertical: 50),
    child: Column(
      children: [
        Icon(
          Icons.inbox_outlined,
          size: 45,
          color: Colors.white.withOpacity(0.20),
        ),
        const SizedBox(height: 10),
        Text(
          text,
          style: GoogleFonts.dmSans(
            color: Colors.white.withOpacity(0.45),
            fontSize: 10,
          ),
        ),
      ],
    ),
  );
}
