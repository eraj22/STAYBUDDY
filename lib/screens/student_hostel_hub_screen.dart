import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';

import '../api.dart';
import '../routes.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'student_qr_attendance_screen.dart';
import 'student_location_sharing_screen.dart';
import 'chatbot_screen.dart';
import 'ai_recommendation_screen.dart';

/// Resident-only tools. This screen receives a confirmed assignment from the
/// access endpoint; every action is then checked again by the server.
class StudentHostelHubScreen extends StatefulWidget {
  const StudentHostelHubScreen({super.key, required this.assignment});

  final Map<String, dynamic> assignment;

  @override
  State<StudentHostelHubScreen> createState() => _StudentHostelHubScreenState();
}

class _StudentHostelHubScreenState extends State<StudentHostelHubScreen> {
  final GlobalKey<ScaffoldState> _portalScaffoldKey = GlobalKey<ScaffoldState>();
  bool _loading = true;
  String? _loadError;
  Map<String, dynamic> _profile = const {};
  Map<String, dynamic> _fees = const {};
  List<dynamic> _announcementItems = const [];
  List<dynamic> _complaints = const [];
  List<dynamic> _attendanceItems = const [];

  int get _hostelId => int.tryParse(widget.assignment['hostel_id'].toString()) ?? 0;
  String get _hostelName => widget.assignment['hostel_name']?.toString() ?? 'My Hostel';
  String get _studentName => _profile['name']?.toString().trim().isNotEmpty == true ? _profile['name'].toString() : 'Student';

  @override
  void initState() { super.initState(); _loadDashboard(); }

  Future<T?> _safe<T>(Future<T> request) async { try { return await request; } catch (_) { return null; } }

  Future<void> _loadDashboard() async {
    if (mounted) setState(() { _loading = true; _loadError = null; });
    try {
      final data = await Future.wait([
        _safe(Api().getStudentProfile()), _safe(Api().getAnnouncements()),
        _safe(Api().getComplaints()), _safe(Api().getStudentGateAttendance()),
        _safe(Api().getStudentFeeStatement()),
      ]);
      if (!mounted) return;
      final allAnnouncements = List<dynamic>.from(data[1] as List? ?? const []);
      setState(() {
        _profile = Map<String, dynamic>.from(data[0] as Map? ?? const {});
        _announcementItems = allAnnouncements.where((item) => Map<String, dynamic>.from(item as Map)['hostel_id'].toString() == '$_hostelId').toList();
        _complaints = List<dynamic>.from(data[2] as List? ?? const []);
        _attendanceItems = List<dynamic>.from(data[3] as List? ?? const []);
        _fees = Map<String, dynamic>.from(data[4] as Map? ?? const {});
        _loading = false;
      });
    } catch (e) { if (mounted) setState(() { _loading = false; _loadError = e.toString().replaceFirst('Exception: ', ''); }); }
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: .55,
          tintColor: AppTheme.tealDeep,
          tintOpacity: .34,
          cardMargin: const EdgeInsets.all(12),
          cardRadius: 30,
          cardColor: const Color(0xE6003331),
          child: SafeArea(child: _loading ? const Center(child: CircularProgressIndicator(color: AppTheme.tealMint)) : _loadError != null ? _loadFailure() : _portal()),
        ),
      ),
    );
  }

  Widget _portal() => Scaffold(
    key: _portalScaffoldKey,
    backgroundColor: Colors.transparent,
    drawer: _studentDrawer(),
    body: RefreshIndicator(
      color: AppTheme.tealMint,
      onRefresh: _loadDashboard,
      child: ListView(padding: const EdgeInsets.fromLTRB(26, 24, 26, 38), children: [
        SizedBox(
          height: 54,
          child: Stack(children: [
            Align(
              alignment: Alignment.centerLeft,
              child: Container(
                width: 54, height: 54, padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16)),
                child: Image.asset('assets/images/logo.png', fit: BoxFit.contain, semanticLabel: 'StayBuddy logo'),
              ),
            ),
            Align(
              alignment: Alignment.center,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 10),
                decoration: BoxDecoration(color: const Color(0x29000000), border: Border.all(color: const Color(0xFF367572)), borderRadius: BorderRadius.circular(23)),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  const Icon(Icons.verified_user_rounded, color: AppTheme.tealMint, size: 16),
                  const SizedBox(width: 7),
                  Text('Student Portal', style: GoogleFonts.dmSans(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w800)),
                ]),
              ),
            ),
            Align(
              alignment: Alignment.centerRight,
              child: InkWell(
                onTap: () => _portalScaffoldKey.currentState?.openDrawer(),
                borderRadius: BorderRadius.circular(28),
                child: Container(width: 52, height: 52, decoration: BoxDecoration(shape: BoxShape.circle, color: const Color(0x29000000), border: Border.all(color: const Color(0xFF367572))), child: const Icon(Icons.menu_rounded, color: Colors.white, size: 27)),
              ),
            ),
          ]),
        ),
        const SizedBox(height: 28), Text(_greeting(), style: GoogleFonts.dmSans(color: const Color(0xFFB7D8D4), fontSize: 13)),
        Row(children: [CircleAvatar(radius: 35, backgroundColor: AppTheme.teal, child: Text(_studentName.substring(0, 1).toUpperCase(), style: GoogleFonts.playfairDisplay(color: Colors.white, fontSize: 31, fontWeight: FontWeight.w700))), const SizedBox(width: 13), Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(_studentName, maxLines: 1, overflow: TextOverflow.ellipsis, style: GoogleFonts.playfairDisplay(color: Colors.white, fontSize: 31, fontWeight: FontWeight.w700)), Text('Here is your stay at $_hostelName', maxLines: 1, overflow: TextOverflow.ellipsis, style: GoogleFonts.dmSans(color: const Color(0xFFB7D8D4), fontSize: 12))]))]),
        const SizedBox(height: 25), _membershipCard(),
        const SizedBox(height: 27), _section('Today at your hostel', 'Your live resident overview'), const SizedBox(height: 13), _liveStats(),
      ]),
    ),
  );

  String _greeting() { final h = DateTime.now().hour; return h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening'; }
  Widget _section(String title, String subtitle) => Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Row(children: [Container(width: 5, height: 30, decoration: BoxDecoration(color: AppTheme.teal, borderRadius: BorderRadius.circular(4))), const SizedBox(width: 11), Text(title, style: GoogleFonts.dmSans(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w800))]), Padding(padding: const EdgeInsets.only(left: 16, top: 2), child: Text(subtitle, style: GoogleFonts.dmSans(color: const Color(0xFF9FC1BE), fontSize: 11)))]);
  Widget _liveStats() { final totals = Map<String, dynamic>.from(_fees['totals'] as Map? ?? const {}); final today = DateTime.now().toIso8601String().substring(0, 10); final scans = _attendanceItems.where((e) => e.toString().contains(today)).length; final issues = _complaints.where((e) { final s = Map<String,dynamic>.from(e as Map)['status']?.toString().toLowerCase(); return s != 'resolved' && s != 'closed'; }).length; return GridView.count(shrinkWrap: true, physics: const NeverScrollableScrollPhysics(), crossAxisCount: 2, crossAxisSpacing: 12, mainAxisSpacing: 12, childAspectRatio: 1.08, children: [_stat(Icons.bed_rounded, 'My room', '${widget.assignment['room_number'] ?? '—'}'), _stat(Icons.qr_code_scanner_rounded, 'Today scans', '$scans'), _stat(Icons.receipt_long_rounded, 'Due balance', 'PKR ${totals['remaining'] ?? 0}'), _stat(Icons.report_problem_outlined, 'Open issues', '$issues')]); }
  Widget _stat(IconData icon, String label, String value) => Container(padding: const EdgeInsets.all(13), decoration: BoxDecoration(color: const Color(0xE603383A), border: Border.all(color: const Color(0x668AD7BF)), borderRadius: BorderRadius.circular(20)), child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisAlignment: MainAxisAlignment.center, children: [Container(width: 37, height: 33, decoration: BoxDecoration(color: const Color(0x1F4ED2AA), borderRadius: BorderRadius.circular(10)), child: Icon(icon, color: AppTheme.tealMint, size: 20)), const SizedBox(height: 8), Text(value, maxLines: 1, overflow: TextOverflow.ellipsis, style: GoogleFonts.dmSans(color: Colors.white, fontSize: 15, fontWeight: FontWeight.w800)), Text(label, style: GoogleFonts.dmSans(color: const Color(0xFFB7D8D4), fontSize: 10))]));
  Widget _studentDrawer() => Drawer(backgroundColor: const Color(0xFF003B39), child: SafeArea(child: ListView(padding: const EdgeInsets.all(14), children: [
    Container(padding: const EdgeInsets.all(18), decoration: BoxDecoration(color: const Color(0xD0002625), border: Border.all(color: const Color(0xFF2D6966)), borderRadius: BorderRadius.circular(24)), child: Row(children: [CircleAvatar(radius: 26, backgroundColor: AppTheme.teal, child: Text(_studentName.substring(0, 1).toUpperCase(), style: GoogleFonts.playfairDisplay(color: Colors.white, fontSize: 24, fontWeight: FontWeight.w700))), const SizedBox(width: 13), Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(_studentName, maxLines: 1, overflow: TextOverflow.ellipsis, style: GoogleFonts.dmSans(color: Colors.white, fontWeight: FontWeight.w800)), Text('Hostel Resident', style: GoogleFonts.dmSans(color: const Color(0xFFB7D8D4), fontSize: 11)), const SizedBox(height: 7), Text(_hostelName, maxLines: 1, overflow: TextOverflow.ellipsis, style: GoogleFonts.dmSans(color: AppTheme.tealMint, fontSize: 11, fontWeight: FontWeight.w700))]))])),
    const SizedBox(height: 16),
    _drawerTile(Icons.grid_view_rounded, 'Dashboard', () {}),
    _drawerTile(Icons.badge_outlined, 'Personal records', () => _personalRecords(context)),
    _drawerTile(Icons.report_problem_outlined, 'Submit complaint', () => _complaint(context)),
    _drawerTile(Icons.qr_code_scanner_rounded, 'Gate attendance', () => _attendance(context)),
    _drawerTile(Icons.swap_horiz_rounded, 'Room & bed change', () => _roomChange(context)),
    _drawerTile(Icons.campaign_rounded, 'Hostel announcements', () => _announcements(context)),
    _drawerTile(Icons.forum_outlined, 'Chats', () => _chatChooser(context)),
    _drawerTile(Icons.receipt_long_rounded, 'Rent & payment receipts', _feesPanel),
    _drawerTile(Icons.share_location_rounded, 'Share my location', () => Navigator.push(context, MaterialPageRoute(builder: (_) => const StudentLocationSharingScreen()))),
    _drawerTile(Icons.auto_awesome_rounded, 'Hostel recommendations', () => Navigator.push(context, MaterialPageRoute(builder: (_) => const AiRecommendationScreen())), smart: true),
    _drawerTile(Icons.smart_toy_outlined, 'AI Assistant', () => Navigator.push(context, MaterialPageRoute(builder: (_) => const ChatbotScreen())), smart: true),
    const SizedBox(height: 14), FilledButton.icon(style: FilledButton.styleFrom(backgroundColor: AppTheme.teal, minimumSize: const Size.fromHeight(53)), onPressed: _logout, icon: const Icon(Icons.logout_rounded), label: const Text('Logout'))
  ])));
  Widget _drawerTile(IconData icon, String label, VoidCallback onTap, {bool smart = false}) => Padding(padding: const EdgeInsets.only(bottom: 8), child: Material(color: const Color(0xB0003030), borderRadius: BorderRadius.circular(15), child: InkWell(onTap: () { Navigator.pop(context); onTap(); }, borderRadius: BorderRadius.circular(15), child: Container(padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 14), decoration: BoxDecoration(border: Border.all(color: smart ? const Color(0xFF25807B) : const Color(0xFF15514E)), borderRadius: BorderRadius.circular(15)), child: Row(children: [Icon(icon, color: smart ? AppTheme.tealMint : const Color(0xFFBED3D0), size: 21), const SizedBox(width: 14), Text(label, style: GoogleFonts.dmSans(color: smart ? AppTheme.tealMint : Colors.white, fontSize: 13, fontWeight: FontWeight.w700))])))));
  void _feesPanel() { final invoices = List<dynamic>.from(_fees['invoices'] as List? ?? const []); final totals = Map<String,dynamic>.from(_fees['totals'] as Map? ?? const {}); _panel(context, _sheetBody('Rent & payment receipts', Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text('Outstanding: PKR ${totals['remaining'] ?? 0}', style: const TextStyle(color: AppTheme.tealMint, fontWeight: FontWeight.w800, fontSize: 17)), const SizedBox(height: 6), const Text('Monthly rent is due on the 5th. Pay online and upload a receipt, or pay cash in person—the warden will mark a cash payment as paid.', style: TextStyle(color: Color(0xFFB7D8D4), fontSize: 12)), const SizedBox(height: 12), Expanded(child: invoices.isEmpty ? const Center(child: Text('No rent or fee records have been issued yet.', style: TextStyle(color: Color(0xFFB7D8D4)))) : ListView.builder(itemCount: invoices.length, itemBuilder: (_, i) { final item = Map<String,dynamic>.from(invoices[i] as Map); final open = '${item['status']}'.toLowerCase() != 'paid' && '${item['remaining_amount'] ?? 0}' != '0'; return Card(color: const Color(0xFF093C3A), child: Padding(padding: const EdgeInsets.all(12), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text('${item['title'] ?? 'Monthly hostel rent'}', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700)), const SizedBox(height: 4), Text('Due ${_shortDate(item['due_date'])} • ${item['status'] ?? 'pending'}', style: const TextStyle(color: Color(0xFFB7D8D4))), const SizedBox(height: 5), Text('PKR ${item['remaining_amount'] ?? item['amount'] ?? 0}', style: const TextStyle(color: AppTheme.tealMint, fontWeight: FontWeight.w800)), if (open && item['id'] != null) Align(alignment: Alignment.centerRight, child: TextButton.icon(onPressed: () => _uploadReceipt(item), icon: const Icon(Icons.upload_file_rounded), label: const Text('Upload receipt')))]))); }))]))); }
  Future<void> _uploadReceipt(Map<String, dynamic> invoice) async {
    final invoiceId = int.tryParse(invoice['id'].toString());
    if (invoiceId == null) return;
    final amount = TextEditingController(text: '${invoice['remaining_amount'] ?? invoice['amount'] ?? ''}');
    XFile? receipt;
    _panel(context, StatefulBuilder(builder: (context, setLocal) => _form('Upload payment receipt', [
      const Text('Upload your payment proof directly. It is saved immediately and sent to the warden for review.', style: TextStyle(color: Color(0xFFB7D8D4))),
      const SizedBox(height: 12), TextField(controller: amount, keyboardType: const TextInputType.numberWithOptions(decimal: true), decoration: const InputDecoration(labelText: 'Amount paid (PKR)')),
      const SizedBox(height: 4), OutlinedButton.icon(onPressed: () async { final picked = await ImagePicker().pickImage(source: ImageSource.gallery, imageQuality: 80, maxWidth: 1800); if (picked != null) setLocal(() => receipt = picked); }, icon: const Icon(Icons.photo_outlined), label: Text(receipt == null ? 'Choose receipt image' : receipt!.name)),
    ], 'Upload receipt', () async {
      if (receipt == null) throw Exception('Choose a receipt image.');
      final paid = double.tryParse(amount.text.trim());
      if (paid == null || paid <= 0) throw Exception('Enter the amount paid.');
      final bytes = await receipt!.readAsBytes();
      if (bytes.length > 5 * 1024 * 1024) throw Exception('Receipt image must be 5 MB or smaller.');
      final mime = receipt!.mimeType == 'image/png' ? 'image/png' : 'image/jpeg';
      await Api().submitStudentPaymentProof(invoiceId: invoiceId, amount: paid, paymentReference: 'Student receipt upload', paidAt: DateTime.now(), filename: receipt!.name, contentType: mime, contentBase64: base64Encode(bytes));
      await _loadDashboard();
    })));
  }
  String _shortDate(dynamic raw) { final parsed = DateTime.tryParse(raw?.toString() ?? ''); if (parsed == null) return '—'; final date = parsed.toLocal(); const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']; return '${date.day.toString().padLeft(2, '0')} ${months[date.month - 1]} ${date.year}'; }
  Widget _loadFailure() => Center(child: Padding(padding: const EdgeInsets.all(28), child: Column(mainAxisSize: MainAxisSize.min, children: [const Icon(Icons.cloud_off_rounded, color: AppTheme.tealMint, size: 48), const SizedBox(height: 12), Text(_loadError ?? 'Unable to load your portal', textAlign: TextAlign.center, style: const TextStyle(color: Colors.white)), const SizedBox(height: 14), FilledButton(onPressed: _loadDashboard, child: const Text('Try again'))])));
  Future<void> _logout() async { await Api().logout(); if (mounted) Navigator.of(context).pushNamedAndRemoveUntil(Routes.login, (_) => false); }

  Widget _membershipCard() => Container(
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(color: AppTheme.tealDeep, borderRadius: BorderRadius.circular(21), border: Border.all(color: AppTheme.tealMint.withOpacity(.45))),
    child: Row(children: [
      const CircleAvatar(backgroundColor: Color(0x334ED2AA), child: Icon(Icons.verified_user_rounded, color: AppTheme.tealMint)),
      const SizedBox(width: 12),
      Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('Verified resident', style: GoogleFonts.dmSans(color: Colors.white, fontWeight: FontWeight.w800)),
        const SizedBox(height: 3),
        Text('Floor ${widget.assignment['floor_label'] ?? '—'}  ·  Room ${widget.assignment['room_number'] ?? '—'}  ·  Bed ${widget.assignment['bed_number'] ?? '—'}', style: GoogleFonts.dmSans(color: Colors.white70, fontSize: 12)),
      ])),
    ]),
  );

  void _attendance(BuildContext context) => Navigator.push(context, MaterialPageRoute(builder: (_) => const StudentQrAttendanceScreen()));

  void _personalRecords(BuildContext context) => _panel(context, FutureBuilder<Map<String, dynamic>>(
    future: Api().getStudentProfile(),
    builder: (_, snapshot) {
      if (!snapshot.hasData) return const Center(child: Padding(padding: EdgeInsets.all(36), child: CircularProgressIndicator(color: AppTheme.teal)));
      final p = snapshot.data!;
      final values = <String, String>{
        'Name': '${p['name'] ?? '—'}', 'CNIC': '${p['student_cnic'] ?? '—'}', 'Email': '${p['email'] ?? '—'}', 'Phone': '${p['phone'] ?? '—'}',
        'University': '${p['university'] ?? '—'}', 'Department': '${p['department'] ?? '—'}', 'Semester': '${p['semester'] ?? '—'}',
        'Hostel room': '${widget.assignment['floor_label'] ?? '—'} / ${widget.assignment['room_number'] ?? '—'} / Bed ${widget.assignment['bed_number'] ?? '—'}',
        'Parent / guardian': '${p['parent_name'] ?? '—'} (${p['parent_relation'] ?? 'Guardian'})', 'Parent phone': '${p['parent_phone'] ?? '—'}', 'Parent email': '${p['parent_email'] ?? '—'}',
      };
      return _sheetBody('Personal records', ListView(shrinkWrap: true, children: [
        ...values.entries.map((e) => ListTile(dense: true, title: Text(e.key, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700)), subtitle: Text(e.value, style: const TextStyle(color: Color(0xFFB7D8D4), fontWeight: FontWeight.w600)))),
        const SizedBox(height: 12), OutlinedButton.icon(onPressed: () { Navigator.pop(context); _editPersonalRecords(p); }, icon: const Icon(Icons.edit_outlined), label: const Text('Add / edit my details')),
      ]));
    },
  ));

  void _editPersonalRecords(Map<String, dynamic> current) {
    final name = TextEditingController(text: current['name']?.toString() ?? '');
    final phone = TextEditingController(text: current['phone']?.toString() ?? '');
    final cnic = TextEditingController(text: current['student_cnic']?.toString() ?? '');
    final university = TextEditingController(text: current['university']?.toString() ?? '');
    final department = TextEditingController(text: current['department']?.toString() ?? '');
    final semester = TextEditingController(text: current['semester']?.toString() ?? '');
    final parentName = TextEditingController(text: current['parent_name']?.toString() ?? '');
    final parentPhone = TextEditingController(text: current['parent_phone']?.toString() ?? '');
    final parentRelation = TextEditingController(text: current['parent_relation']?.toString() ?? '');
    final parentEmail = TextEditingController(text: current['parent_email']?.toString() ?? '');
    _panel(context, _form('Edit personal records', [
      TextField(controller: name, decoration: const InputDecoration(labelText: 'Name')), TextField(controller: phone, decoration: const InputDecoration(labelText: 'Phone number')), TextField(controller: cnic, decoration: const InputDecoration(labelText: 'CNIC')),
      TextField(controller: university, decoration: const InputDecoration(labelText: 'University')), TextField(controller: department, decoration: const InputDecoration(labelText: 'Department')), TextField(controller: semester, decoration: const InputDecoration(labelText: 'Semester')),
      const SizedBox(height: 8), const Text('Parent / guardian', style: TextStyle(color: AppTheme.tealMint, fontWeight: FontWeight.w800)), TextField(controller: parentName, decoration: const InputDecoration(labelText: 'Name')), TextField(controller: parentRelation, decoration: const InputDecoration(labelText: 'Relationship')), TextField(controller: parentPhone, decoration: const InputDecoration(labelText: 'Phone number')), TextField(controller: parentEmail, decoration: const InputDecoration(labelText: 'Email')),
    ], 'Save details', () async {
      await Api().updateStudentProfile({'name': name.text, 'phone': phone.text, 'student_cnic': cnic.text, 'university': university.text, 'department': department.text, 'semester': semester.text, 'parent_name': parentName.text, 'parent_relation': parentRelation.text, 'parent_phone': parentPhone.text, 'parent_email': parentEmail.text});
      await _loadDashboard();
    }));
  }

  void _bedAvailability(BuildContext context) => _panel(context, FutureBuilder<Map<String, dynamic>>(
    future: Api().getBedAvailability(_hostelId),
    builder: (_, snapshot) {
      if (snapshot.hasError) return _sheetBody('Bed availability', _error(snapshot.error));
      if (!snapshot.hasData) return const SizedBox(height: 180, child: Center(child: CircularProgressIndicator(color: AppTheme.teal)));
      final rooms = List<dynamic>.from(snapshot.data!['rooms'] as List? ?? const []);
      return _sheetBody('Live bed availability', ListView.builder(shrinkWrap: true, itemCount: rooms.length, itemBuilder: (_, i) {
        final room = Map<String, dynamic>.from(rooms[i] as Map);
        final beds = List<dynamic>.from(room['beds'] as List? ?? const []);
        return Card(child: Padding(padding: const EdgeInsets.all(12), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('Floor ${room['floor_label']}  ·  Room ${room['room_number']}', style: const TextStyle(fontWeight: FontWeight.w800, color: Colors.white)),
          Text('${room['room_type']} · ${room['available_capacity']}/${room['capacity']} beds free', style: const TextStyle(color: Color(0xFFB7D8D4))),
          const SizedBox(height: 7), Wrap(spacing: 6, runSpacing: 6, children: beds.map((item) { final b = Map<String, dynamic>.from(item as Map); final free = b['status'].toString() == 'available'; return Chip(label: Text('Bed ${b['bed_number']}'), backgroundColor: free ? const Color(0xFF14574F) : const Color(0xFF254446), labelStyle: const TextStyle(color: Colors.white), side: BorderSide.none); }).toList()),
        ])));
      }));
    },
  ));

  void _announcements(BuildContext context) => _panel(context, _sheetBody('Hostel announcements', _announcementItems.isEmpty ? const Padding(padding: EdgeInsets.all(25), child: Center(child: Text('No announcements yet.'))) : ListView.builder(shrinkWrap: true, itemCount: _announcementItems.length, itemBuilder: (_, i) { final a = Map<String, dynamic>.from(_announcementItems[i] as Map); return ListTile(leading: const Icon(Icons.campaign_rounded, color: AppTheme.teal), title: Text('${a['title'] ?? 'Announcement'}'), subtitle: Text('${a['body'] ?? ''}')); })));

  Future<void> _roomChange(BuildContext context) async {
    final reason = TextEditingController();
    final data = await Future.wait([_safe(Api().getBedAvailability(_hostelId)), _safe(Api().getStudentRoomChangeRequests())]);
    final response = data[0] as Map<String, dynamic>?;
    final requests = List<dynamic>.from(data[1] as List? ?? const []);
    if (!context.mounted) return;
    final rooms = List<dynamic>.from(response?['rooms'] as List? ?? const []).where((item) {
      final room = Map<String, dynamic>.from(item as Map);
      return List<dynamic>.from(room['beds'] as List? ?? const []).any((bed) => Map<String, dynamic>.from(bed as Map)['status'] == 'available');
    }).toList();
    String? roomId;
    String? bedNumber;
    _panel(context, StatefulBuilder(builder: (context, setLocal) {
      final selected = roomId == null ? null : rooms.cast<Map>().map((item) => Map<String, dynamic>.from(item)).firstWhere((item) => item['id'].toString() == roomId, orElse: () => <String, dynamic>{});
      final beds = selected == null ? const <dynamic>[] : List<dynamic>.from(selected['beds'] as List? ?? const []).where((bed) => Map<String, dynamic>.from(bed as Map)['status'] == 'available').toList();
      return _form('Request room / bed change', [
        const Text('Choose any currently empty bed. The warden receives your selection and decides whether to approve it.', style: TextStyle(color: Color(0xFFB7D8D4))),
        const SizedBox(height: 12),
        DropdownButtonFormField<String>(value: roomId, isExpanded: true, decoration: const InputDecoration(labelText: 'Available room'), items: rooms.map((item) { final room = Map<String, dynamic>.from(item as Map); return DropdownMenuItem(value: room['id'].toString(), child: Text('Floor ${room['floor_label']} · Room ${room['room_number']}')); }).toList(), onChanged: (value) => setLocal(() { roomId = value; bedNumber = null; })),
        const SizedBox(height: 12),
        DropdownButtonFormField<String>(value: bedNumber, isExpanded: true, decoration: const InputDecoration(labelText: 'Available bed'), items: beds.map((item) { final bed = Map<String, dynamic>.from(item as Map); return DropdownMenuItem(value: bed['bed_number'].toString(), child: Text('Bed ${bed['bed_number']}')); }).toList(), onChanged: roomId == null ? null : (value) => setLocal(() => bedNumber = value)),
        const SizedBox(height: 12), TextField(controller: reason, maxLines: 4, maxLength: 2000, decoration: const InputDecoration(labelText: 'Reason for the request')),
        if (requests.isNotEmpty) ...[
          const SizedBox(height: 14), const Divider(color: Color(0xFF2D6966)), const Text('My room / bed requests', style: TextStyle(color: Colors.white, fontWeight: FontWeight.w800)),
          ...requests.take(5).map((item) { final request = Map<String, dynamic>.from(item as Map); final target = request['room_number'] == null ? 'Selected room / bed' : 'Floor ${request['floor_label']} · Room ${request['room_number']} · Bed ${request['requested_bed_number']}'; return ListTile(dense: true, contentPadding: EdgeInsets.zero, title: Text('Request from ${_shortDate(request['created_at'])}', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700)), subtitle: Text('$target\n${request['reason'] ?? ''}', maxLines: 2, overflow: TextOverflow.ellipsis, style: const TextStyle(color: Color(0xFFB7D8D4))), trailing: Text('${request['status'] ?? 'pending'}'.toUpperCase(), style: const TextStyle(color: AppTheme.tealMint, fontWeight: FontWeight.w800, fontSize: 10))); }),
        ],
      ], 'Send request to warden', () async {
        if (roomId == null || bedNumber == null) throw Exception('Choose an available room and bed.');
        if (reason.text.trim().length < 3) throw Exception('Please enter a reason of at least 3 characters.');
        await Api().createRoomChangeRequest(hostelId: _hostelId, reason: reason.text, requestedRoomId: int.parse(roomId!), requestedBedNumber: bedNumber!); await _loadDashboard();
      });
    }));
  }

  void _complaint(BuildContext context) {
    final description = TextEditingController(); final images = <XFile>[];
    _panel(context, StatefulBuilder(builder: (context, setLocal) => _form('Submit a complaint', [
      const Text('Describe the issue and add up to three images. The warden will review its status.', style: TextStyle(color: Color(0xFFB7D8D4))),
      const SizedBox(height: 10), TextField(controller: description, maxLines: 5, maxLength: 4000, decoration: const InputDecoration(hintText: 'Describe the issue clearly')),
      Wrap(spacing: 8, children: [
        OutlinedButton.icon(onPressed: images.length >= 3 ? null : () async { final picked = await ImagePicker().pickMultiImage(imageQuality: 65, maxWidth: 1280); setLocal(() => images.addAll(picked.take(3 - images.length))); }, icon: const Icon(Icons.add_photo_alternate_outlined), label: Text('Add images (${images.length}/3)')),
        ...images.asMap().entries.map((entry) => InputChip(label: Text('Image ${entry.key + 1}'), onDeleted: () => setLocal(() => images.removeAt(entry.key)))),
      ]),
      if (_complaints.isNotEmpty) ...[
        const SizedBox(height: 14), const Divider(color: Color(0xFF2D6966)), const SizedBox(height: 8), const Text('My previous complaints', style: TextStyle(color: Colors.white, fontWeight: FontWeight.w800)),
        ..._complaints.take(6).map((item) { final complaint = Map<String, dynamic>.from(item as Map); final status = complaint['status']?.toString() ?? 'submitted'; return Container(margin: const EdgeInsets.only(top: 8), padding: const EdgeInsets.all(11), decoration: BoxDecoration(color: const Color(0xFF07393C), border: Border.all(color: const Color(0xFF2D6966)), borderRadius: BorderRadius.circular(13)), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Row(children: [Expanded(child: Text('${complaint['description'] ?? 'Complaint'}', maxLines: 2, overflow: TextOverflow.ellipsis, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700))), Text(status.toUpperCase(), style: const TextStyle(color: AppTheme.tealMint, fontWeight: FontWeight.w800, fontSize: 10))]), const SizedBox(height: 4), Text('Submitted ${_shortDate(complaint['created_at'])}', style: const TextStyle(color: Color(0xFFB7D8D4), fontSize: 11))])); }),
      ],
    ], 'Submit complaint', () async {
      if (description.text.trim().length < 3) throw Exception('Describe the issue in at least 3 characters.');
      final encoded = <String>[]; for (final image in images) { final bytes = await image.readAsBytes(); if (bytes.length > 1500000) throw Exception('Each image must be 1.5 MB or smaller.'); encoded.add('data:image/jpeg;base64,${base64Encode(bytes)}'); }
      await Api().fileComplaint(hostelId: _hostelId, category: 'general', severity: 'medium', description: description.text, attachments: encoded); await _loadDashboard();
    })));
  }

  void _chatChooser(BuildContext context) => _panel(context, _sheetBody('Choose a chat', Column(mainAxisSize: MainAxisSize.min, children: [
    _chatTile(context, 'parents', 'Chat with parents', Icons.family_restroom_rounded),
    _chatTile(context, 'warden', 'Chat with warden', Icons.support_agent_rounded),
  ])));

  Widget _chatTile(BuildContext context, String channel, String label, IconData icon) => ListTile(leading: Icon(icon, color: AppTheme.teal), title: Text(label), trailing: const Icon(Icons.chevron_right), onTap: () { Navigator.pop(context); Navigator.push(context, MaterialPageRoute(builder: (_) => _StudentChatScreen(hostelId: _hostelId, hostelName: _hostelName, channel: channel, label: label))); });

  void _panel(BuildContext context, Widget child) => showModalBottomSheet<void>(context: context, isScrollControlled: true, backgroundColor: Colors.transparent, builder: (_) => Theme(data: ThemeData.dark().copyWith(colorScheme: const ColorScheme.dark(primary: AppTheme.tealMint, surface: Color(0xFF003B39), onSurface: Colors.white), inputDecorationTheme: InputDecorationTheme(filled: true, fillColor: const Color(0xFF062B2B), hintStyle: GoogleFonts.dmSans(color: const Color(0xFF9FC1BE)), labelStyle: GoogleFonts.dmSans(color: const Color(0xFFB7D8D4)), enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(16), borderSide: const BorderSide(color: Color(0x668AD7BF))), focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(16), borderSide: const BorderSide(color: AppTheme.tealMint)))), child: DraggableScrollableSheet(initialChildSize: .77, minChildSize: .42, maxChildSize: .92, expand: false, builder: (_, controller) => Container(decoration: const BoxDecoration(color: Color(0xFF003B39), borderRadius: BorderRadius.vertical(top: Radius.circular(28))), child: child))));
  Widget _sheetBody(String title, Widget body) => Padding(padding: const EdgeInsets.fromLTRB(20, 12, 20, 25), child: Column(children: [Container(width: 42, height: 4, decoration: BoxDecoration(color: const Color(0xFF6E9693), borderRadius: BorderRadius.circular(8))), const SizedBox(height: 17), Align(alignment: Alignment.centerLeft, child: Text(title, style: GoogleFonts.playfairDisplay(fontSize: 23, fontWeight: FontWeight.w700, color: Colors.white))), const SizedBox(height: 12), Expanded(child: body)]));
  Widget _error(Object? error) => Center(child: Text(error.toString().replaceFirst('Exception: ', ''), textAlign: TextAlign.center, style: const TextStyle(color: Color(0xFFB7D8D4))));
  Widget _form(String title, List<Widget> fields, String label, Future<void> Function() submit) => _HubForm(title: title, fields: fields, actionLabel: label, submit: submit);
}

class _HubForm extends StatefulWidget { const _HubForm({required this.title, required this.fields, required this.actionLabel, required this.submit}); final String title; final List<Widget> fields; final String actionLabel; final Future<void> Function() submit; @override State<_HubForm> createState() => _HubFormState(); }
class _HubFormState extends State<_HubForm> { bool _saving = false; String? _error; @override Widget build(BuildContext context) => Padding(padding: EdgeInsets.fromLTRB(20, 12, 20, 24 + MediaQuery.of(context).viewInsets.bottom), child: SingleChildScrollView(child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [Container(width: 42, height: 4, alignment: Alignment.center, decoration: BoxDecoration(color: const Color(0xFF6E9693), borderRadius: BorderRadius.circular(4))), const SizedBox(height: 18), Text(widget.title, style: GoogleFonts.playfairDisplay(fontSize: 24, fontWeight: FontWeight.w700, color: Colors.white)), const SizedBox(height: 16), ...widget.fields, if (_error != null) Padding(padding: const EdgeInsets.only(top: 6), child: Text(_error!, style: const TextStyle(color: Color(0xFFFF9B94)))), const SizedBox(height: 14), ElevatedButton(style: ElevatedButton.styleFrom(backgroundColor: AppTheme.teal, foregroundColor: Colors.white, padding: const EdgeInsets.symmetric(vertical: 14)), onPressed: _saving ? null : () async { setState(() { _saving = true; _error = null; }); try { await widget.submit(); if (mounted) { Navigator.pop(context); ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Saved successfully.'))); } } catch (e) { if (mounted) setState(() => _error = e.toString().replaceFirst('Exception: ', '')); } finally { if (mounted) setState(() => _saving = false); } }, child: _saving ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)) : Text(widget.actionLabel)), ]))); }

class _StudentChatScreen extends StatefulWidget { const _StudentChatScreen({required this.hostelId, required this.hostelName, required this.channel, required this.label}); final int hostelId; final String hostelName; final String channel; final String label; @override State<_StudentChatScreen> createState() => _StudentChatScreenState(); }
class _StudentChatScreenState extends State<_StudentChatScreen> { final _body = TextEditingController(); bool _loading = true; bool _sending = false; String? _error; List<dynamic> _messages = const []; @override void initState() { super.initState(); _load(); } Future<void> _load() async { try { final response = await Api().getHostelChat(hostelId: widget.hostelId, channel: widget.channel); if (mounted) setState(() { _messages = List<dynamic>.from(response['messages'] as List? ?? const []); _loading = false; }); } catch (e) { if (mounted) setState(() { _error = e.toString().replaceFirst('Exception: ', ''); _loading = false; }); } } Future<void> _send() async { if (_body.text.trim().isEmpty || _sending) return; setState(() => _sending = true); try { await Api().sendHostelChat(hostelId: widget.hostelId, channel: widget.channel, body: _body.text); _body.clear(); await _load(); } finally { if (mounted) setState(() => _sending = false); } } @override void dispose() { _body.dispose(); super.dispose(); } @override Widget build(BuildContext context) => Scaffold(appBar: AppBar(title: Text(widget.label), actions: [IconButton(onPressed: _load, icon: const Icon(Icons.refresh_rounded))]), body: Column(children: [Expanded(child: _loading ? const Center(child: CircularProgressIndicator(color: AppTheme.teal)) : _error != null ? Center(child: Text(_error!)) : _messages.isEmpty ? const Center(child: Text('No messages yet. Start this conversation.')) : ListView.builder(padding: const EdgeInsets.all(14), itemCount: _messages.length, itemBuilder: (_, i) { final m = Map<String, dynamic>.from(_messages[i] as Map); return Align(alignment: Alignment.centerLeft, child: Container(margin: const EdgeInsets.only(bottom: 8), padding: const EdgeInsets.all(11), decoration: BoxDecoration(color: AppTheme.teal.withOpacity(.09), borderRadius: BorderRadius.circular(14)), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text('${m['sender_name'] ?? 'Resident'}', style: const TextStyle(fontWeight: FontWeight.w800, color: AppTheme.tealDeep)), const SizedBox(height: 3), Text('${m['body'] ?? ''}')]))); })), SafeArea(top: false, child: Padding(padding: const EdgeInsets.all(10), child: Row(children: [Expanded(child: TextField(controller: _body, minLines: 1, maxLines: 4, decoration: const InputDecoration(hintText: 'Write a message'))), const SizedBox(width: 8), IconButton.filled(onPressed: _sending ? null : _send, icon: const Icon(Icons.send_rounded))])))])); }
