import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';

import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';

/// Warden-facing QR gate console. The QR itself is generated and persisted by
/// the backend; this screen never stores the secret locally.
class WardenQrAttendanceScreen extends StatefulWidget {
  final String hostelName;

  const WardenQrAttendanceScreen({super.key, required this.hostelName});

  @override
  State<WardenQrAttendanceScreen> createState() =>
      _WardenQrAttendanceScreenState();
}

class _WardenQrAttendanceScreenState extends State<WardenQrAttendanceScreen> {
  bool _loading = true;
  bool _creating = false;
  bool _downloading = false;
  String? _error;
  int? _hostelId;
  Map<String, dynamic>? _dashboard;
  Map<String, dynamic>? _generated;
  List<dynamic> _gates = const [];
  DateTime? _selectedDate;
  bool _datePickerVisible = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final assigned = await Api().getWardenAssignedHostel();
      final hostel = Map<String, dynamic>.from(assigned['hostel'] as Map);
      final id = int.parse(hostel['id'].toString());
      final values = await Future.wait([
        Api().getGateQrs(id),
        Api().getQrAttendanceDashboard(id, date: _selectedDateKey),
      ]);
      if (!mounted) return;
      setState(() {
        _hostelId = id;
        _gates = values[0] as List<dynamic>;
        _dashboard = values[1] as Map<String, dynamic>;
        Map<String, dynamic>? activeGate;
        for (final rawGate in _gates) {
          final gate = Map<String, dynamic>.from(rawGate as Map);
          if (gate['active'] == true) {
            activeGate = gate;
            break;
          }
        }
        if (_generated == null && activeGate?['qr_image'] != null) {
          _generated = {'qr_image': activeGate!['qr_image']};
        }
        _error = null;
      });
    } catch (error) {
      if (mounted)
        setState(
          () => _error = error.toString().replaceFirst('Exception: ', ''),
        );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _generate() async {
    if (_hostelId == null) return;
    setState(() => _creating = true);
    try {
      final created = await Api().createGateQr(hostelId: _hostelId!);
      if (!mounted) return;
      setState(() => _generated = created);
      await _load();
    } catch (error) {
      if (mounted)
        _message(error.toString().replaceFirst('Exception: ', ''), error: true);
    } finally {
      if (mounted) setState(() => _creating = false);
    }
  }

  void _message(String text, {bool error = false}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(text),
        backgroundColor: error ? const Color(0xFF963C3C) : AppTheme.tealDark,
      ),
    );
  }

  Future<Uint8List> _posterPdf() async {
    final image = _generated?['qr_image']?.toString();
    if (image == null || !image.contains(','))
      throw StateError('No QR image is available to download.');
    final bytes = base64Decode(image.split(',').last);
    final document = pw.Document();
    document.addPage(
      pw.Page(
        pageFormat: PdfPageFormat.a4,
        build: (_) => pw.Center(
          child: pw.Column(
            mainAxisSize: pw.MainAxisSize.min,
            children: [
              pw.Text(
                widget.hostelName,
                style: pw.TextStyle(
                  fontSize: 26,
                  fontWeight: pw.FontWeight.bold,
                ),
              ),
              pw.SizedBox(height: 12),
              pw.Text(
                'Scan this QR with the StayBuddy app to record gate attendance.',
              ),
              pw.SizedBox(height: 24),
              pw.Image(pw.MemoryImage(bytes), width: 330, height: 330),
              pw.SizedBox(height: 18),
              pw.Text(
                'QR attendance is available only within the hostel location area.',
              ),
            ],
          ),
        ),
      ),
    );
    return document.save();
  }

  Future<void> _downloadQrPoster() async {
    setState(() => _downloading = true);
    try {
      final safeName = widget.hostelName
          .replaceAll(RegExp(r'[^A-Za-z0-9]+'), '-')
          .replaceAll(RegExp(r'^-|-$'), '');
      await Printing.sharePdf(
        bytes: await _posterPdf(),
        filename: '${safeName.isEmpty ? 'staybuddy' : safeName}-gate-qr.pdf',
      );
      if (mounted)
        _message(
          'QR poster download is ready. You can print the saved PDF any time.',
        );
    } catch (error) {
      if (mounted)
        _message(
          'Could not download the QR poster: ${error.toString()}',
          error: true,
        );
    } finally {
      if (mounted) setState(() => _downloading = false);
    }
  }

  Color get _surface => const Color(0xE603383A);
  Color get _border => const Color(0x4D8AD7BF);

  String? get _selectedDateKey {
    final date = _selectedDate;
    if (date == null) return null;
    return '${date.year.toString().padLeft(4, '0')}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')}';
  }

  void _selectDate() {
    setState(() => _datePickerVisible = !_datePickerVisible);
  }

  void _applyDate(DateTime date) {
    setState(() {
      _selectedDate = date;
      _datePickerVisible = false;
    });
    _load();
  }

  void _clearDate() {
    if (_selectedDate == null) return;
    setState(() {
      _selectedDate = null;
      _datePickerVisible = false;
    });
    _load();
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
          tintOpacity: .34,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: _loading
                ? const Center(
                    child: CircularProgressIndicator(color: AppTheme.tealMint),
                  )
                : _error != null
                ? _errorView()
                : RefreshIndicator(
                    color: AppTheme.tealMint,
                    onRefresh: _load,
                    child: ListView(
                      padding: const EdgeInsets.fromLTRB(20, 12, 20, 32),
                      children: [
                        _top(),
                        const SizedBox(height: 22),
                        _summary(),
                        const SizedBox(height: 20),
                        _qrPanel(),
                      ],
                    ),
                  ),
          ),
        ),
      ),
    );
  }

  Widget _top() => Row(
    children: [
      _roundButton(Icons.arrow_back_rounded, () => Navigator.pop(context)),
      const SizedBox(width: 12),
      Expanded(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'QR Attendance',
              style: GoogleFonts.playfairDisplay(
                color: Colors.white,
                fontSize: 28,
                fontWeight: FontWeight.w700,
              ),
            ),
            Text(
              widget.hostelName,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.dmSans(
                color: const Color(0xBFD0E8E5),
                fontSize: 12,
              ),
            ),
          ],
        ),
      ),
    ],
  );

  Widget _roundButton(IconData icon, VoidCallback onTap) => InkWell(
    onTap: onTap,
    borderRadius: BorderRadius.circular(50),
    child: Container(
      width: 44,
      height: 44,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: Colors.white.withOpacity(.1),
        border: Border.all(color: _border),
      ),
      child: Icon(icon, color: Colors.white),
    ),
  );

  Widget _summary() {
    final summary = Map<String, dynamic>.from(
      _dashboard?['summary'] as Map? ?? const {},
    );
    return Row(
      children: [
        _stat('Inside', summary['inside'] ?? 0, Icons.home_rounded),
        const SizedBox(width: 10),
        _stat(
          'Outside',
          summary['outside'] ?? 0,
          Icons.directions_walk_rounded,
        ),
      ],
    );
  }

  Widget _stat(String label, dynamic value, IconData icon) => Expanded(
    child: Container(
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: _surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: AppTheme.tealMint, size: 20),
          const SizedBox(height: 10),
          Text(
            '$value',
            style: GoogleFonts.dmSans(
              color: Colors.white,
              fontSize: 22,
              fontWeight: FontWeight.w700,
            ),
          ),
          Text(
            label,
            style: GoogleFonts.dmSans(
              color: const Color(0xBFD0E8E5),
              fontSize: 10,
            ),
          ),
        ],
      ),
    ),
  );

  DateTime? _eventMoment(Map<String, dynamic> event) {
    final raw = event['occurred_at']?.toString();
    if (raw == null || raw.isEmpty) return null;
    return DateTime.tryParse(raw)?.toLocal();
  }

  String _eventDate(DateTime? time) {
    if (time == null) return 'Date unavailable';
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
    return '${time.day.toString().padLeft(2, '0')} ${months[time.month - 1]} ${time.year}';
  }

  String _eventTime(DateTime? time) {
    if (time == null) return 'Time unavailable';
    final hour = time.hour % 12 == 0 ? 12 : time.hour % 12;
    final minute = time.minute.toString().padLeft(2, '0');
    return '$hour:$minute ${time.hour >= 12 ? 'PM' : 'AM'}';
  }

  Widget _attendanceLog() {
    final allEvents = List<dynamic>.from(
      _dashboard?['recent_events'] as List? ?? const [],
    );
    final selected = _selectedDate;
    final events = selected == null
        ? allEvents
        : allEvents.where((raw) {
            final time = _eventMoment(Map<String, dynamic>.from(raw as Map));
            return time != null &&
                time.year == selected.year &&
                time.month == selected.month &&
                time.day == selected.day;
          }).toList();
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: _surface,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: _border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.fact_check_rounded, color: AppTheme.tealMint),
              const SizedBox(width: 9),
              Expanded(
                child: Text(
                  'Latest attendance',
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 17,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              IconButton(
                onPressed: _load,
                tooltip: 'Refresh attendance',
                color: AppTheme.tealMint,
                icon: const Icon(Icons.refresh_rounded),
              ),
            ],
          ),
          Text(
            'IN/OUT scans saved with server date and time',
            style: GoogleFonts.dmSans(
              color: const Color(0xBFD0E8E5),
              fontSize: 11,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _selectDate,
                  icon: const Icon(Icons.calendar_month_rounded, size: 18),
                  label: Text(
                    _selectedDate == null
                        ? 'Select date'
                        : _eventDate(_selectedDate),
                  ),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: AppTheme.tealMint,
                    side: const BorderSide(color: AppTheme.tealMint),
                    padding: const EdgeInsets.symmetric(vertical: 11),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14),
                    ),
                  ),
                ),
              ),
              if (_selectedDate != null) ...[
                const SizedBox(width: 8),
                IconButton(
                  onPressed: _clearDate,
                  tooltip: 'Show all dates',
                  color: Colors.white70,
                  icon: const Icon(Icons.clear_rounded),
                ),
              ],
            ],
          ),
          if (_datePickerVisible) ...[
            const SizedBox(height: 12),
            _inlineDatePicker(),
          ],
          const SizedBox(height: 14),
          if (events.isEmpty)
            _emptyAttendanceLog(hasSelectedDate: selected != null)
          else
            ...events
                .take(12)
                .map(
                  (raw) => _attendanceEventTile(
                    Map<String, dynamic>.from(raw as Map),
                  ),
                ),
        ],
      ),
    );
  }

  Widget _inlineDatePicker() {
    final today = DateTime.now();
    return Container(
      padding: const EdgeInsets.all(6),
      decoration: BoxDecoration(
        color: const Color(0xFF07383A),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0x668AD7BF)),
      ),
      child: Theme(
        data: Theme.of(context).copyWith(
          colorScheme: const ColorScheme.dark(
            primary: AppTheme.tealMint,
            onPrimary: AppTheme.tealDeep,
            surface: Color(0xFF07383A),
            onSurface: Colors.white,
          ),
        ),
        child: CalendarDatePicker(
          initialDate: _selectedDate ?? today,
          firstDate: DateTime(2020),
          lastDate: today.add(const Duration(days: 1)),
          onDateChanged: _applyDate,
        ),
      ),
    );
  }

  Widget _emptyAttendanceLog({bool hasSelectedDate = false}) => Container(
    width: double.infinity,
    padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 14),
    decoration: BoxDecoration(
      color: const Color(0x33075B57),
      borderRadius: BorderRadius.circular(16),
    ),
    child: Text(
      hasSelectedDate
          ? 'No Attendance Record'
          : 'No attendance scans recorded yet.',
      textAlign: TextAlign.center,
      style: GoogleFonts.dmSans(color: const Color(0xBFD0E8E5), fontSize: 12),
    ),
  );

  Widget _attendanceEventTile(Map<String, dynamic> event) {
    final isIn = event['direction']?.toString().toLowerCase() == 'in';
    final time = _eventMoment(event);
    final location =
        [
              event['floor_label']?.toString(),
              event['room_number'] == null
                  ? null
                  : 'Room ${event['room_number']}',
              event['bed_number'] == null ? null : 'Bed ${event['bed_number']}',
            ]
            .whereType<String>()
            .where((value) => value.trim().isNotEmpty)
            .join('  •  ');
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: isIn ? const Color(0x220FA88A) : const Color(0x222B93A6),
        borderRadius: BorderRadius.circular(17),
        border: Border.all(
          color: isIn ? const Color(0x9950D6B6) : const Color(0x995AA8B7),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: isIn ? AppTheme.teal : const Color(0xFF256C6E),
              borderRadius: BorderRadius.circular(13),
            ),
            child: Icon(
              isIn ? Icons.login_rounded : Icons.logout_rounded,
              color: Colors.white,
              size: 21,
            ),
          ),
          const SizedBox(width: 11),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        event['student_name']?.toString() ?? 'Student',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.dmSans(
                          color: Colors.white,
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      isIn ? 'IN' : 'OUT',
                      style: GoogleFonts.dmSans(
                        color: isIn
                            ? AppTheme.tealMint
                            : const Color(0xFF9BD9E4),
                        fontSize: 11,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 3),
                Text(
                  '${_eventDate(time)}  •  ${_eventTime(time)}',
                  style: GoogleFonts.dmSans(
                    color: const Color(0xBFD0E8E5),
                    fontSize: 11,
                  ),
                ),
                if (location.isNotEmpty) ...[
                  const SizedBox(height: 7),
                  Text(
                    location,
                    style: GoogleFonts.dmSans(
                      color: Colors.white70,
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _qrPanel() {
    final image = _generated?['qr_image']?.toString();
    final bytes = image != null && image.contains(',')
        ? base64Decode(image.split(',').last)
        : null;
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: _surface,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: _border),
      ),
      child: Column(
        children: [
          const Icon(
            Icons.qr_code_2_rounded,
            color: AppTheme.tealMint,
            size: 30,
          ),
          const SizedBox(height: 8),
          Text(
            'Post gate QR code',
            style: GoogleFonts.dmSans(
              color: Colors.white,
              fontSize: 17,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 5),
          Text(
            'Students scan this at the hostel gate to record IN or OUT with server time.',
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(
              color: const Color(0xBFD0E8E5),
              fontSize: 11,
            ),
          ),
          if (bytes != null) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(10),
              color: Colors.white,
              child: Image.memory(bytes, width: 190, height: 190),
            ),
            const SizedBox(height: 10),
            Text(
              'Generated now — replacing this QR revokes the previous one.',
              textAlign: TextAlign.center,
              style: GoogleFonts.dmSans(color: AppTheme.tealMint, fontSize: 10),
            ),
          ],
          if (bytes == null &&
              _gates.any(
                (gate) =>
                    Map<String, dynamic>.from(gate as Map)['active'] == true,
              )) ...[
            const SizedBox(height: 10),
            Text(
              'This active QR was created before downloads were enabled. Replace it once to save and download it later.',
              textAlign: TextAlign.center,
              style: GoogleFonts.dmSans(
                color: const Color(0xBFD0E8E5),
                fontSize: 10,
              ),
            ),
          ],
          const SizedBox(height: 17),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: _creating ? null : _generate,
              icon: _creating
                  ? const SizedBox(
                      width: 17,
                      height: 17,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : const Icon(Icons.qr_code_rounded),
              label: Text(
                _generated == null
                    ? 'Generate Main Gate QR'
                    : 'Replace QR code',
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppTheme.teal,
                foregroundColor: Colors.white,
              ),
            ),
          ),
          if (bytes != null) ...[
            const SizedBox(height: 10),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: _downloading ? null : _downloadQrPoster,
                icon: _downloading
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: AppTheme.tealMint,
                        ),
                      )
                    : const Icon(Icons.download_rounded),
                label: Text(
                  _downloading ? 'Preparing download…' : 'Download QR Code',
                ),
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppTheme.tealMint,
                  side: const BorderSide(color: AppTheme.tealMint),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _errorView() => Center(
    child: Padding(
      padding: const EdgeInsets.all(28),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.cloud_off_rounded, color: Colors.white, size: 46),
          const SizedBox(height: 14),
          Text(
            'QR attendance is unavailable',
            style: GoogleFonts.playfairDisplay(
              color: Colors.white,
              fontSize: 22,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 7),
          Text(
            _error!,
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(
              color: const Color(0xBFD0E8E5),
              fontSize: 11,
            ),
          ),
          const SizedBox(height: 16),
          ElevatedButton(onPressed: _load, child: const Text('Try again')),
        ],
      ),
    ),
  );
}
