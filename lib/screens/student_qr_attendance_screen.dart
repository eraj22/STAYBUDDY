import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:mobile_scanner/mobile_scanner.dart';

import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';

/// Student gate flow. A scan is sent directly to the attendance API, which
/// writes the IN/OUT event and authoritative server timestamp in PostgreSQL.
class StudentQrAttendanceScreen extends StatefulWidget {
  const StudentQrAttendanceScreen({super.key});

  @override
  State<StudentQrAttendanceScreen> createState() =>
      _StudentQrAttendanceScreenState();
}

class _StudentQrAttendanceScreenState extends State<StudentQrAttendanceScreen> {
  final _scanner = MobileScannerController();
  bool _scanning = false;
  bool _submitting = false;
  bool _checkingLocation = true;
  bool _insideHostel = false;
  String? _gateCode;
  String? _error;
  String? _locationStatus;
  String? _hostelName;
  List<dynamic> _history = const [];

  String? get _lastDirection => _history.isEmpty
      ? null
      : Map<String, dynamic>.from(
          _history.first as Map,
        )['direction']?.toString().toLowerCase();

  @override
  void initState() {
    super.initState();
    _loadHistory();
    _checkHostelLocation();
  }

  @override
  void dispose() {
    _scanner.dispose();
    super.dispose();
  }

  Future<void> _loadHistory() async {
    try {
      final history = await Api().getStudentGateAttendance();
      if (mounted) setState(() => _history = history);
    } catch (_) {
      // The scanner remains usable even if history cannot be loaded.
    }
  }

  String? _extractGateCode(String raw) {
    final value = raw.trim();
    if (RegExp(r'^[A-Za-z0-9_-]{43}$').hasMatch(value)) return value;
    try {
      final uri = Uri.parse(value);
      final code = uri.fragment
          .split('&')
          .firstWhere((item) => item.startsWith('code='), orElse: () => '')
          .replaceFirst('code=', '');
      return RegExp(r'^[A-Za-z0-9_-]{43}$').hasMatch(code) ? code : null;
    } catch (_) {
      return null;
    }
  }

  void _onDetect(BarcodeCapture capture) {
    if (_submitting || _gateCode != null) return;
    for (final barcode in capture.barcodes) {
      final code = _extractGateCode(barcode.rawValue ?? '');
      if (code != null) {
        setState(() {
          _gateCode = code;
          _scanning = false;
          _error = null;
        });
        _scanner.stop();
        return;
      }
    }
    setState(() => _error = 'This is not a StayBuddy gate QR code.');
  }

  Future<Position?> _currentPosition() async {
    if (!await Geolocator.isLocationServiceEnabled()) return null;
    var permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }
    if (permission == LocationPermission.denied ||
        permission == LocationPermission.deniedForever)
      return null;
    return Geolocator.getCurrentPosition(
      desiredAccuracy: LocationAccuracy.high,
    );
  }

  bool _isInsideBoundary(
    double latitude,
    double longitude,
    List<dynamic> rawBoundary,
  ) {
    final points = rawBoundary
        .whereType<Map>()
        .map(
          (point) => (
            latitude: (point['latitude'] as num?)?.toDouble(),
            longitude: (point['longitude'] as num?)?.toDouble(),
          ),
        )
        .where((point) => point.latitude != null && point.longitude != null)
        .toList();
    if (points.length < 3) return false;
    var inside = false;
    for (
      var current = 0, previous = points.length - 1;
      current < points.length;
      previous = current++
    ) {
      final a = points[current];
      final b = points[previous];
      final crosses = (a.latitude! > latitude) != (b.latitude! > latitude);
      final intersection =
          (b.longitude! - a.longitude!) *
              (latitude - a.latitude!) /
              (b.latitude! - a.latitude!) +
          a.longitude!;
      if (crosses && longitude < intersection) inside = !inside;
    }
    return inside;
  }

  Future<void> _checkHostelLocation() async {
    setState(() => _checkingLocation = true);
    try {
      final access = await Api().getStudentQrAttendanceAccess();
      final hostel = Map<String, dynamic>.from(access['hostel'] as Map);
      final position = await _currentPosition();
      if (position == null) {
        if (mounted)
          setState(() {
            _insideHostel = false;
            _hostelName = hostel['name']?.toString();
            _locationStatus = 'Allow location permission to use QR attendance.';
          });
        return;
      }
      final radius = (access['radius_m'] as num?)?.toDouble() ?? 200;
      final boundary = access['attendance_boundary'] as List? ?? const [];
      final distance = Geolocator.distanceBetween(
        position.latitude,
        position.longitude,
        (hostel['latitude'] as num).toDouble(),
        (hostel['longitude'] as num).toDouble(),
      );
      if (mounted)
        setState(() {
          _insideHostel = boundary.length >= 3
              ? _isInsideBoundary(
                  position.latitude,
                  position.longitude,
                  boundary,
                )
              : distance <= radius + position.accuracy.clamp(0, 50);
          _hostelName = hostel['name']?.toString();
          _locationStatus = _insideHostel
              ? 'You are within the hostel QR attendance boundary.'
              : boundary.length >= 3
              ? 'Move inside the hostel building, gate, or entrance boundary to enable scanning.'
              : 'Move within ${radius.round()}m of the hostel to enable scanning.';
        });
    } catch (error) {
      if (mounted)
        setState(() {
          _insideHostel = false;
          _locationStatus = error.toString().replaceFirst('Exception: ', '');
        });
    } finally {
      if (mounted) setState(() => _checkingLocation = false);
    }
  }

  Future<void> _record(String direction) async {
    final code = _gateCode;
    if (code == null) return;
    setState(() => _submitting = true);
    try {
      final location = await _currentPosition();
      final result = await Api().submitGateAttendance(
        gateCode: code,
        direction: direction,
        latitude: location?.latitude,
        longitude: location?.longitude,
        accuracyMeters: location?.accuracy,
      );
      if (!mounted) return;
      final event = Map<String, dynamic>.from(
        result['event'] as Map? ?? const {},
      );
      final time = event['occurred_at']?.toString() ?? 'now';
      _show('${direction.toUpperCase()} recorded at $time');
      setState(() => _gateCode = null);
      await _loadHistory();
    } catch (error) {
      if (mounted)
        setState(
          () => _error = error.toString().replaceFirst('Exception: ', ''),
        );
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  void _show(String text) => ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(content: Text(text), backgroundColor: AppTheme.tealDark),
  );

  void _startScanner() {
    if (!_insideHostel) {
      _show(
        _locationStatus ?? 'QR scanning is available only inside the hostel.',
      );
      return;
    }
    setState(() {
      _gateCode = null;
      _error = null;
      _scanning = true;
    });
    _scanner.start();
  }

  @override
  Widget build(BuildContext context) => ResponsiveShell(
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
          child: ListView(
            padding: const EdgeInsets.fromLTRB(20, 12, 20, 35),
            children: [
              Row(
                children: [
                  InkWell(
                    onTap: () => Navigator.pop(context),
                    borderRadius: BorderRadius.circular(30),
                    child: Container(
                      width: 44,
                      height: 44,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.white.withOpacity(.1),
                      ),
                      child: const Icon(
                        Icons.arrow_back_rounded,
                        color: Colors.white,
                      ),
                    ),
                  ),
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
                          _hostelName ?? 'Checking hostel location…',
                          style: GoogleFonts.dmSans(
                            color: const Color(0xBFD0E8E5),
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 24),
              _locationCard(),
              const SizedBox(height: 14),
              _scannerCard(),
              if (_error != null)
                Padding(
                  padding: const EdgeInsets.only(top: 12),
                  child: Text(
                    _error!,
                    textAlign: TextAlign.center,
                    style: GoogleFonts.dmSans(
                      color: const Color(0xFFFFB4AB),
                      fontSize: 11,
                    ),
                  ),
                ),
              const SizedBox(height: 22),
              Text(
                'Recent gate activity',
                style: GoogleFonts.playfairDisplay(
                  color: Colors.white,
                  fontSize: 21,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 10),
              if (_history.isEmpty)
                _emptyHistory()
              else
                ..._history.take(5).map(_historyTile),
            ],
          ),
        ),
      ),
    ),
  );

  Widget _scannerCard() => Container(
    padding: const EdgeInsets.all(18),
    decoration: _card(),
    child: Column(
      children: [
        if (_scanning)
          ClipRRect(
            borderRadius: BorderRadius.circular(18),
            child: SizedBox(
              height: 260,
              child: MobileScanner(controller: _scanner, onDetect: _onDetect),
            ),
          )
        else if (_gateCode != null) ...[
          const Icon(
            Icons.verified_rounded,
            size: 50,
            color: AppTheme.tealMint,
          ),
          const SizedBox(height: 12),
          Text(
            'Gate QR verified',
            style: GoogleFonts.dmSans(
              color: Colors.white,
              fontSize: 17,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            'Choose the direction you are travelling. The time is saved by the server.',
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(
              color: const Color(0xBFD0E8E5),
              fontSize: 11,
            ),
          ),
          const SizedBox(height: 17),
          Row(
            children: [
              Expanded(
                child: SizedBox(
                  height: 52,
                  child: _directionButton(
                    'Mark OUT',
                    Icons.logout_rounded,
                    'out',
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: SizedBox(
                  height: 52,
                  child: _directionButton('Mark IN', Icons.login_rounded, 'in'),
                ),
              ),
            ],
          ),
        ] else ...[
          Icon(
            Icons.qr_code_scanner_rounded,
            size: 58,
            color: _insideHostel ? AppTheme.tealMint : const Color(0xFF8BA7A4),
          ),
          const SizedBox(height: 12),
          Text(
            'Ready to scan',
            style: GoogleFonts.dmSans(
              color: Colors.white,
              fontSize: 17,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            'Use your phone camera at the hostel gate. Location is requested only for this scan and sent with your consent.',
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(
              color: const Color(0xBFD0E8E5),
              fontSize: 11,
            ),
          ),
          const SizedBox(height: 17),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: _checkingLocation || !_insideHostel
                  ? null
                  : _startScanner,
              icon: const Icon(Icons.camera_alt_rounded),
              label: Text(
                _checkingLocation
                    ? 'Checking location…'
                    : _insideHostel
                    ? 'Scan gate QR'
                    : 'Scanning disabled',
              ),
            ),
          ),
        ],
      ],
    ),
  );

  Widget _directionButton(String label, IconData icon, String value) {
    final isIn = value == 'in';
    final isRecorded = _lastDirection == value;
    return ElevatedButton(
      onPressed: _submitting || _lastDirection == value
          ? null
          : () => _record(value),
      style: ElevatedButton.styleFrom(
        backgroundColor: isIn ? AppTheme.teal : const Color(0xFF256C6E),
        foregroundColor: Colors.white,
        disabledBackgroundColor: const Color(0xFF315D5E),
        disabledForegroundColor: Colors.white70,
        minimumSize: Size.zero,
        padding: const EdgeInsets.symmetric(horizontal: 10),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      ),
      child: _submitting
          ? const SizedBox(
              width: 18,
              height: 18,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: Colors.white,
              ),
            )
          : Row(
              mainAxisAlignment: MainAxisAlignment.center,
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, size: 19),
                const SizedBox(width: 7),
                Flexible(
                  child: Text(
                    isRecorded ? 'Recorded' : label,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    softWrap: false,
                    style: GoogleFonts.dmSans(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ],
            ),
    );
  }

  BoxDecoration _card() => BoxDecoration(
    color: const Color(0xE603383A),
    borderRadius: BorderRadius.circular(24),
    border: Border.all(color: const Color(0x4D8AD7BF)),
  );

  Widget _locationCard() => Container(
    padding: const EdgeInsets.all(13),
    decoration: BoxDecoration(
      color: _insideHostel ? const Color(0xCC07534E) : const Color(0xCC3A4241),
      borderRadius: BorderRadius.circular(16),
      border: Border.all(
        color: _insideHostel
            ? AppTheme.tealMint.withOpacity(.6)
            : const Color(0xFF78908D),
      ),
    ),
    child: Row(
      children: [
        Icon(
          _checkingLocation
              ? Icons.location_searching_rounded
              : _insideHostel
              ? Icons.location_on_rounded
              : Icons.location_off_rounded,
          color: _insideHostel ? AppTheme.tealMint : Colors.white70,
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                _insideHostel ? 'QR scanning active' : 'QR scanning disabled',
                style: GoogleFonts.dmSans(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                  fontSize: 12,
                ),
              ),
              Text(
                _locationStatus ?? 'Checking your location…',
                style: GoogleFonts.dmSans(
                  color: const Color(0xBFD0E8E5),
                  fontSize: 10,
                ),
              ),
            ],
          ),
        ),
        IconButton(
          onPressed: _checkingLocation ? null : _checkHostelLocation,
          icon: const Icon(Icons.refresh_rounded, color: Colors.white),
          tooltip: 'Check location again',
        ),
      ],
    ),
  );

  Widget _emptyHistory() => Container(
    padding: const EdgeInsets.all(18),
    decoration: _card(),
    child: Text(
      'No QR attendance has been recorded yet.',
      style: GoogleFonts.dmSans(color: const Color(0xBFD0E8E5), fontSize: 11),
    ),
  );

  Widget _historyTile(dynamic raw) {
    final item = Map<String, dynamic>.from(raw as Map);
    final direction = item['direction']?.toString().toUpperCase() ?? '—';
    final time = item['submitted_at']?.toString().replaceFirst('T', ' ') ?? '';
    final inside = direction == 'IN';
    return Container(
      margin: const EdgeInsets.only(bottom: 9),
      padding: const EdgeInsets.all(13),
      decoration: _card(),
      child: Row(
        children: [
          Icon(
            inside ? Icons.login_rounded : Icons.logout_rounded,
            color: inside ? AppTheme.tealMint : const Color(0xFFFFCC80),
          ),
          const SizedBox(width: 11),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Recorded $direction',
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                    fontSize: 12,
                  ),
                ),
                Text(
                  time,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: const Color(0xBFD0E8E5),
                    fontSize: 10,
                  ),
                ),
              ],
            ),
          ),
          Text(
            item['status']?.toString() ?? '',
            style: GoogleFonts.dmSans(color: AppTheme.tealMint, fontSize: 10),
          ),
        ],
      ),
    );
  }
}
