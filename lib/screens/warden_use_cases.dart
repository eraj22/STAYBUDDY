// ═══════════════════════════════════════════════════════════════════
// WARDEN USE CASES — ALL 11 SCREENS
// ═══════════════════════════════════════════════════════════════════
// 1.  View & Manage Complaints
// 2.  View All Students' Attendance
// 3.  Make Announcements
// 4.  Assign Penalties
// 5.  Handle Room Allocation
// 6.  View Student & Parent Details
// 7.  Notify About Empty Beds
// 8.  Chat with Parents & Students
// 9.  Update Cleaning Schedule
// 10. Receive AI Complaint Categories   ← integrated with /categorize
// 11. Get AI Auto-Suggestions           ← integrated with /categorize
// ═══════════════════════════════════════════════════════════════════

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import '../theme.dart';
import '../widgets/responsive_shell.dart';

// ── Shared helpers ─────────────────────────────────────────────────
AppBar _appBar(BuildContext ctx, String title, {List<Widget>? actions}) =>
    AppBar(
      title: Text(title),
      leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded),
          onPressed: () => Navigator.pop(ctx)),
      actions: actions,
    );

Widget _field(TextEditingController c, String hint, IconData icon,
    {int lines = 1, TextInputType? type}) =>
    Container(
      decoration: BoxDecoration(
          color: AppTheme.card, borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppTheme.border, width: 0.5)),
      child: TextField(
        controller: c, maxLines: lines, keyboardType: type,
        style: GoogleFonts.dmSans(fontSize: 14, color: AppTheme.textPrimary),
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 13),
          prefixIcon: Icon(icon, size: 18, color: AppTheme.textMuted),
          border: InputBorder.none,
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 14, vertical: 13)),
      ),
    );

Widget _badge(String t, Color bg, Color fg) => Container(
  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
  decoration:
      BoxDecoration(color: bg, borderRadius: BorderRadius.circular(6)),
  child: Text(t,
      style: GoogleFonts.dmSans(
          fontSize: 10, fontWeight: FontWeight.w600, color: fg)),
);


// ══════════════════════════════════════════════════════════════════
// 1. COMPLAINTS SCREEN  (AI-powered: Use Case 10 & 11)
// ══════════════════════════════════════════════════════════════════
class WardenComplaintsScreen extends StatefulWidget {
  final String hostelName;
  const WardenComplaintsScreen({super.key, required this.hostelName});
  @override
  State<WardenComplaintsScreen> createState() => _WardenComplaintsScreenState();
}

class _WardenComplaintsScreenState extends State<WardenComplaintsScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabs;

  final _complaints = <Map<String, dynamic>>[
    {'id':'C001','student':'Sara Khan',  'room':'102','text':'WiFi has been down for 3 days.',      'category':'Maintenance',     'subcategory':'Other Maintenance','priority':'High',  'status':'Pending',    'suggestion':'Assign technician to fix the issue within 24 hours.','confidence':0.80,'time':'2 hrs ago'},
    {'id':'C002','student':'Amna Raza',  'room':'201','text':'Bathroom not cleaned for 2 days.',   'category':'Cleanliness',     'subcategory':'Bathroom',        'priority':'Medium','status':'In Progress','suggestion':'Schedule housekeeping to clean the area immediately.', 'confidence':0.91,'time':'5 hrs ago'},
    {'id':'C003','student':'Hina Malik', 'room':'301','text':'Security guard absent at midnight.', 'category':'Safety/Security', 'subcategory':'Security Guard',  'priority':'High',  'status':'Pending',    'suggestion':'Investigate immediately; if serious call police.',   'confidence':0.85,'time':'Yesterday'},
    {'id':'C004','student':'Fatima Ali', 'room':'103','text':'Food quality is very bad.',          'category':'Food Quality',    'subcategory':'Hygiene',         'priority':'Low',   'status':'Resolved',   'suggestion':'Inspect kitchen and food storage; check hygiene.',   'confidence':0.72,'time':'2 days ago'},
    {'id':'C005','student':'Zara Ahmed', 'room':'202','text':'Loud music after midnight.',         'category':'Noise/Disturbance','subcategory':'Night Noise',    'priority':'Medium','status':'Pending',    'suggestion':'Warn the noisy party and enforce quiet hours.',      'confidence':0.88,'time':'3 days ago'},
  ];

  final _textCtrl    = TextEditingController();
  final _studentCtrl = TextEditingController();
  final _roomCtrl    = TextEditingController();

  bool _analyzing = false;
  Map<String, dynamic>? _aiResult;

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 4, vsync: this);
  }

  @override
  void dispose() {
    _tabs.dispose();
    _textCtrl.dispose(); _studentCtrl.dispose(); _roomCtrl.dispose();
    super.dispose();
  }

  // ── Colors ────────────────────────────────────────────────────────
  Color _pColor(String p) => p=='High'
      ? Colors.red.shade700
      : p=='Medium' ? Colors.orange.shade700 : AppTheme.green;

  Color _sColor(String s) {
    switch (s) {
      case 'Resolved':    return AppTheme.green;
      case 'In Progress': return const Color(0xFF185FA5);
      default:            return Colors.orange.shade700;
    }
  }

  // ── AI Call ───────────────────────────────────────────────────────
  Future<void> _analyzeWithAI() async {
    if (_textCtrl.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('Please enter complaint text first', style: GoogleFonts.dmSans()),
        backgroundColor: Colors.red.shade600, behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        margin: const EdgeInsets.all(16)));
      return;
    }
    setState(() { _analyzing = true; _aiResult = null; });
    try {
      final res = await http.post(
        Uri.parse('http://127.0.0.1:8001/categorize'),
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: jsonEncode({'text': _textCtrl.text.trim()}),
      ).timeout(const Duration(seconds: 15));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body) as Map<String, dynamic>;
        if (data['success'] == true) {
          setState(() { _analyzing = false; _aiResult = data; });
        } else {
          setState(() { _analyzing = false; });
          if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(
            content: Text('AI error: ${data['error'] ?? 'Unknown'}', style: GoogleFonts.dmSans()),
            backgroundColor: Colors.red.shade600, behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            margin: const EdgeInsets.all(16)));
        }
      } else {
        setState(() { _analyzing = false; });
        if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Server error ${res.statusCode} — using offline mode', style: GoogleFonts.dmSans()),
          backgroundColor: Colors.orange.shade700, behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          margin: const EdgeInsets.all(16)));
        _offlineFallback();
      }
    } catch (e) {
      setState(() { _analyzing = false; });
      // Show what the actual error is
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('Cannot reach AI server (${e.toString().split(':').first}) — using offline mode',
            style: GoogleFonts.dmSans()),
        backgroundColor: Colors.orange.shade700, behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        margin: const EdgeInsets.all(16)));
      _offlineFallback();
    }
  }

  void _offlineFallback() {
    final t = _textCtrl.text.toLowerCase();
    String cat = 'Other', sub = 'General', pri = 'Medium';
    String sug = 'Review student request and respond within 48 hours.';
    double conf = 0.60;
    if (RegExp(r'wifi|fan|broken|water|electric|light|geyser|ac|pipe|leak|door|window|furniture').hasMatch(t)) {
      cat='Maintenance';    sub='Other Maintenance'; pri='High';   sug='Assign technician to fix the issue within 24 hours.'; conf=0.75;
    } else if (RegExp(r'dirty|clean|bathroom|cockroach|mosquito|pest|smell|toilet|hygiene').hasMatch(t)) {
      cat='Cleanliness';    sub='Bathroom';          pri='Medium'; sug='Schedule housekeeping to clean the area immediately.'; conf=0.82;
    } else if (RegExp(r'food|meal|taste|cook|bread|rotten|poison|portion').hasMatch(t)) {
      cat='Food Quality';   sub='Hygiene';           pri='Low';    sug='Inspect kitchen and food storage; check hygiene.'; conf=0.70;
    } else if (RegExp(r'security|stolen|missing|guard|unauthorized|theft|cctv').hasMatch(t)) {
      cat='Safety/Security';sub='Security Guard';    pri='High';   sug='Investigate immediately; if serious call police.'; conf=0.88;
    } else if (RegExp(r'noise|loud|music|night|disturb|party').hasMatch(t)) {
      cat='Noise/Disturbance';sub='Night Noise';     pri='Medium'; sug='Warn the noisy party and enforce quiet hours.'; conf=0.79;
    } else if (RegExp(r'warden|staff|rude|behavior|attitude|shout').hasMatch(t)) {
      cat='Staff Behavior'; sub='Warden';            pri='Medium'; sug='Counsel the staff member; involve owner if repeated.'; conf=0.73;
    } else if (RegExp(r'roommate|room mate|smoke|sharing|personal space|guest').hasMatch(t)) {
      cat='Roommate Issue'; sub='Personal Space';    pri='Medium'; sug='Mediate between roommates; offer room change if needed.'; conf=0.77;
    } else if (RegExp(r'bill|charge|rent|money|refund|fee|overcharge').hasMatch(t)) {
      cat='Billing/Payment Dispute'; sub='Overcharging'; pri='Medium'; sug='Review charges with account; provide receipt.'; conf=0.71;
    }
    setState(() {
      _analyzing = false;
      _aiResult = {
        'success': true, 'offline': true,
        'category': cat, 'subcategory': sub, 'priority': pri,
        'confidence': conf, 'suggestion': sug,
        'top3_categories': [
          {'category': cat,   'confidence': conf},
          {'category': 'Other', 'confidence': 0.15},
        ],
        'all_suggestions': [sug],
      };
    });
  }

  // ── Add to list ───────────────────────────────────────────────────
  void _addComplaint() {
    if (_aiResult == null || _textCtrl.text.trim().isEmpty) return;
    final r = _aiResult!;
    setState(() {
      _complaints.insert(0, {
        'id':          'C${(_complaints.length + 1).toString().padLeft(3, '0')}',
        'student':     _studentCtrl.text.trim().isNotEmpty ? _studentCtrl.text.trim() : 'Unknown',
        'room':        _roomCtrl.text.trim().isNotEmpty ? _roomCtrl.text.trim() : '?',
        'text':        _textCtrl.text.trim(),
        'category':    r['category'],
        'subcategory': r['subcategory'] ?? 'General',
        'priority':    r['priority'],
        'status':      'Pending',
        'suggestion':  r['suggestion'],
        'confidence':  r['confidence'],
        'time':        'Just now',
      });
      _textCtrl.clear(); _studentCtrl.clear(); _roomCtrl.clear(); _aiResult = null;
    });
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text('Complaint added successfully', style: GoogleFonts.dmSans()),
      backgroundColor: AppTheme.green, behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      margin: const EdgeInsets.all(16)));
    _tabs.animateTo(0);
  }

  // ── Pattern alert ─────────────────────────────────────────────────
  String? get _patternAlert {
    final counts = <String, int>{};
    for (final c in _complaints.where((c) => c['status'] != 'Resolved')) {
      final cat = c['category'] as String;
      counts[cat] = (counts[cat] ?? 0) + 1;
    }
    for (final e in counts.entries) {
      if (e.value >= 2) return '${e.value} "${e.key}" complaints — possible systemic issue';
    }
    return null;
  }

  // ── Build ─────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    final pending  = _complaints.where((c) => c['status'] == 'Pending').length;
    final resolved = _complaints.where((c) => c['status'] == 'Resolved').length;

    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        appBar: AppBar(
          title: const Text('Complaints'),
          leading: IconButton(
              icon: const Icon(Icons.arrow_back_rounded),
              onPressed: () => Navigator.pop(context)),
          bottom: TabBar(
            controller: _tabs,
            labelColor: Colors.white,
            unselectedLabelColor: Colors.white60,
            indicatorColor: AppTheme.tealMint,
            tabs: [
              Tab(text: 'All (${_complaints.length})'),
              Tab(text: 'Pending ($pending)'),
              const Tab(text: 'Add + AI'),
              Tab(text: 'Resolved ($resolved)'),
            ],
          ),
        ),
        body: Column(children: [
          // Stats bar
          Container(
            color: AppTheme.tealDeep,
            padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 24),
            child: Row(mainAxisAlignment: MainAxisAlignment.spaceAround, children: [
              _statChip('Total',   _complaints.length,                                             Colors.white),
              _statChip('High',    _complaints.where((c) => c['priority'] == 'High').length,       Colors.red.shade300),
              _statChip('Pending', pending,                                                         Colors.orange.shade300),
              _statChip('Resolved',resolved,                                                        AppTheme.tealMint),
            ]),
          ),
          // Pattern alert
          if (_patternAlert != null)
            Container(
              margin: const EdgeInsets.fromLTRB(16, 8, 16, 0),
              padding: const EdgeInsets.all(11),
              decoration: BoxDecoration(
                color: Colors.orange.shade50,
                borderRadius: BorderRadius.circular(10),
                border: Border(left: BorderSide(color: Colors.orange.shade600, width: 4))),
              child: Row(children: [
                Icon(Icons.warning_amber_rounded, color: Colors.orange.shade700, size: 16),
                const SizedBox(width: 8),
                Expanded(child: Text('⚠️ Pattern detected: $_patternAlert',
                    style: GoogleFonts.dmSans(fontSize: 12, color: Colors.orange.shade800))),
              ])),
          Expanded(child: TabBarView(controller: _tabs, children: [
            _buildList(_complaints),
            _buildList(_complaints.where((c) => c['status'] == 'Pending').toList()),
            _buildAddTab(),
            _buildList(_complaints.where((c) => c['status'] == 'Resolved').toList()),
          ])),
        ]),
      ),
    );
  }

  Widget _statChip(String label, int n, Color c) => Column(children: [
    Text('$n', style: GoogleFonts.dmSans(color: c, fontSize: 18, fontWeight: FontWeight.w700)),
    Text(label, style: GoogleFonts.dmSans(color: Colors.white60, fontSize: 11)),
  ]);

  // ── Complaint list ─────────────────────────────────────────────────
  Widget _buildList(List<Map<String, dynamic>> items) {
    if (items.isEmpty) return Center(
        child: Text('No complaints', style: GoogleFonts.dmSans(color: AppTheme.textMuted)));
    return ListView.builder(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 20),
      itemCount: items.length,
      itemBuilder: (_, i) => GestureDetector(
        onTap: () => _showDetail(items[i]),
        child: Container(
          margin: const EdgeInsets.only(bottom: 10),
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(color: AppTheme.card,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppTheme.border, width: 0.5)),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              Expanded(child: Text(items[i]['student'] as String,
                  style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w700, color: AppTheme.textPrimary))),
              _badge(items[i]['priority'] as String,
                  _pColor(items[i]['priority'] as String).withOpacity(0.1),
                  _pColor(items[i]['priority'] as String)),
              const SizedBox(width: 6),
              _badge(items[i]['status'] as String,
                  _sColor(items[i]['status'] as String).withOpacity(0.1),
                  _sColor(items[i]['status'] as String)),
            ]),
            const SizedBox(height: 6),
            Row(children: [
              _badge(items[i]['category'] as String, AppTheme.teal.withOpacity(0.08), AppTheme.teal),
              const SizedBox(width: 6),
              _badge('Room ${items[i]['room']}', AppTheme.bgSecondary, AppTheme.textSecondary),
              if (items[i]['subcategory'] != null) ...[ const SizedBox(width: 6),
                _badge(items[i]['subcategory'] as String, const Color(0xFFE3EEF9), const Color(0xFF185FA5))],
              const Spacer(),
              if (items[i]['confidence'] != null)
                Text('AI ${((items[i]['confidence'] as double) * 100).round()}%',
                    style: GoogleFonts.dmSans(fontSize: 10, color: AppTheme.textMuted)),
            ]),
            const SizedBox(height: 8),
            Text(items[i]['text'] as String, maxLines: 2, overflow: TextOverflow.ellipsis,
                style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textSecondary)),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(9),
              decoration: BoxDecoration(color: const Color(0xFFE3EEF9),
                  borderRadius: BorderRadius.circular(8),
                  border: const Border(left: BorderSide(color: Color(0xFF185FA5), width: 2))),
              child: Row(children: [
                const Icon(Icons.lightbulb_rounded, color: Color(0xFF185FA5), size: 12),
                const SizedBox(width: 6),
                Expanded(child: Text(items[i]['suggestion'] as String,
                    maxLines: 1, overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.dmSans(fontSize: 11, color: const Color(0xFF185FA5)))),
              ])),
            const SizedBox(height: 6),
            Text(items[i]['time'] as String,
                style: GoogleFonts.dmSans(fontSize: 10, color: AppTheme.textMuted)),
          ]),
        ),
      ),
    );
  }

  // ── Add + AI tab ───────────────────────────────────────────────────
  Widget _buildAddTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        // Info banner
        Container(
          padding: const EdgeInsets.all(13),
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.06),
            borderRadius: BorderRadius.circular(12),
            border: Border(left: BorderSide(color: AppTheme.teal, width: 3))),
          child: Row(children: [
            const Icon(Icons.auto_awesome_rounded, color: AppTheme.teal, size: 16),
            const SizedBox(width: 10),
            Expanded(child: Text(
              'AI categorizes the complaint (Use Case 10) and suggests '
              'a resolution from complaint history (Use Case 11). '
              'Server: http://127.0.0.1:8001',
              style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.teal, height: 1.4))),
          ])),
        const SizedBox(height: 14),

        // Student + Room fields
        Row(children: [
          Expanded(child: _field(_studentCtrl, 'Student Name', Icons.person_rounded)),
          const SizedBox(width: 10),
          Expanded(child: _field(_roomCtrl, 'Room No.', Icons.meeting_room_rounded)),
        ]),
        const SizedBox(height: 10),

        // Complaint text
        Container(
          decoration: BoxDecoration(color: AppTheme.card,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppTheme.border, width: 0.5)),
          child: TextField(
            controller: _textCtrl, maxLines: 4,
            style: GoogleFonts.dmSans(fontSize: 14, color: AppTheme.textPrimary),
            decoration: InputDecoration(
              hintText: 'Describe the complaint in detail...',
              hintStyle: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 13),
              prefixIcon: const Padding(padding: EdgeInsets.only(bottom: 60),
                  child: Icon(Icons.report_outlined, size: 18, color: AppTheme.textMuted)),
              border: InputBorder.none,
              contentPadding: const EdgeInsets.all(16)),
          )),
        const SizedBox(height: 12),

        // Analyze button
        SizedBox(width: double.infinity, height: 50,
          child: ElevatedButton(
            onPressed: _analyzing ? null : _analyzeWithAI,
            style: ElevatedButton.styleFrom(backgroundColor: AppTheme.teal,
                foregroundColor: Colors.white, elevation: 0,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14))),
            child: _analyzing
                ? Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                    const SizedBox(width: 18, height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)),
                    const SizedBox(width: 10),
                    Text('Analyzing with AI...', style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w600)),
                  ])
                : Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                    const Icon(Icons.auto_awesome_rounded, size: 18),
                    const SizedBox(width: 8),
                    Text('Analyze with AI', style: GoogleFonts.dmSans(fontSize: 15, fontWeight: FontWeight.w600)),
                  ]),
          )),

        // ── AI Result ───────────────────────────────────────────────
        if (_aiResult != null) ...[ const SizedBox(height: 16),
          Container(
            decoration: BoxDecoration(color: AppTheme.card,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: AppTheme.teal.withOpacity(0.3))),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              // Header
              Container(
                padding: const EdgeInsets.fromLTRB(14, 13, 14, 13),
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(0.06),
                  borderRadius: const BorderRadius.vertical(top: Radius.circular(16))),
                child: Row(children: [
                  const Icon(Icons.auto_awesome_rounded, color: AppTheme.teal, size: 18),
                  const SizedBox(width: 8),
                  Text('AI Analysis Result', style: GoogleFonts.dmSans(
                      fontSize: 14, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
                  const Spacer(),
                  // Priority badge
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: _pColor(_aiResult!['priority'] as String).withOpacity(0.12),
                      borderRadius: BorderRadius.circular(8)),
                    child: Row(mainAxisSize: MainAxisSize.min, children: [
                      Icon(Icons.flag_rounded, size: 12,
                          color: _pColor(_aiResult!['priority'] as String)),
                      const SizedBox(width: 4),
                      Text(_aiResult!['priority'] as String,
                          style: GoogleFonts.dmSans(fontSize: 11, fontWeight: FontWeight.w700,
                              color: _pColor(_aiResult!['priority'] as String))),
                    ])),
                  if (_aiResult!['offline'] == true) ...[ const SizedBox(width: 8),
                    Container(padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(color: Colors.grey.shade100, borderRadius: BorderRadius.circular(6)),
                      child: Text('Offline', style: GoogleFonts.dmSans(fontSize: 10, color: Colors.grey.shade600)))],
                ])),

              Padding(
                padding: const EdgeInsets.all(14),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [

                  // Category + Subcategory + Confidence row
                  Row(children: [
                    Expanded(child: _resultBox('Category',
                        _aiResult!['category'] as String, AppTheme.teal)),
                    const SizedBox(width: 8),
                    Expanded(child: _resultBox('Subcategory',
                        _aiResult!['subcategory'] as String? ?? 'General',
                        const Color(0xFF185FA5))),
                    const SizedBox(width: 8),
                    _resultBox('Confidence',
                        '${((_aiResult!['confidence'] as double) * 100).round()}%',
                        AppTheme.green),
                  ]),
                  const SizedBox(height: 12),

                  // AI Suggested Action
                  Container(
                    padding: const EdgeInsets.all(13),
                    decoration: BoxDecoration(color: const Color(0xFFE3EEF9),
                        borderRadius: BorderRadius.circular(12),
                        border: const Border(left: BorderSide(color: Color(0xFF185FA5), width: 3))),
                    child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      const Icon(Icons.lightbulb_rounded, color: Color(0xFF185FA5), size: 16),
                      const SizedBox(width: 10),
                      Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Text('AI Suggested Action', style: GoogleFonts.dmSans(
                            fontSize: 11, fontWeight: FontWeight.w700, color: const Color(0xFF0D4F8C))),
                        const SizedBox(height: 4),
                        Text(_aiResult!['suggestion'] as String, style: GoogleFonts.dmSans(
                            fontSize: 13, color: const Color(0xFF185FA5), height: 1.4)),
                      ])),
                    ])),
                  const SizedBox(height: 12),

                  // Top 3 categories
                  if (_aiResult!['top3_categories'] != null) ...[ 
                    Text('Top Predicted Categories',
                        style: GoogleFonts.dmSans(fontSize: 11, fontWeight: FontWeight.w700, color: AppTheme.textMuted)),
                    const SizedBox(height: 8),
                    ...(_aiResult!['top3_categories'] as List).map((t) {
                      final cat  = t['category'] as String;
                      final conf = ((t['confidence'] as num) * 100).round();
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 6),
                        child: Row(children: [
                          Expanded(child: Text(cat,
                              style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textPrimary))),
                          SizedBox(width: 120,
                            child: ClipRRect(borderRadius: BorderRadius.circular(3),
                              child: LinearProgressIndicator(
                                value: (t['confidence'] as num).toDouble().clamp(0.0, 1.0),
                                backgroundColor: AppTheme.bgSecondary,
                                color: AppTheme.teal, minHeight: 6))),
                          const SizedBox(width: 8),
                          Text('$conf%', style: GoogleFonts.dmSans(
                              fontSize: 11, fontWeight: FontWeight.w700, color: AppTheme.teal)),
                        ]));
                    }),
                    const SizedBox(height: 12),
                  ],

                  // All suggestions (if more than one)
                  if ((_aiResult!['all_suggestions'] as List?)?.length != null &&
                      ((_aiResult!['all_suggestions'] as List).length) > 1) ...[ 
                    Text('Other Suggested Actions',
                        style: GoogleFonts.dmSans(fontSize: 11, fontWeight: FontWeight.w700, color: AppTheme.textMuted)),
                    const SizedBox(height: 6),
                    ...(_aiResult!['all_suggestions'] as List).skip(1).map((s) =>
                      Padding(padding: const EdgeInsets.only(bottom: 5),
                        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                          Container(width: 5, height: 5, margin: const EdgeInsets.only(top: 5, right: 8),
                              decoration: BoxDecoration(color: AppTheme.teal, shape: BoxShape.circle)),
                          Expanded(child: Text(s.toString(),
                              style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textSecondary))),
                        ]))),
                    const SizedBox(height: 12),
                  ],

                  // Add button
                  SizedBox(width: double.infinity, height: 46,
                    child: ElevatedButton(
                      onPressed: _addComplaint,
                      style: ElevatedButton.styleFrom(
                          backgroundColor: AppTheme.green, foregroundColor: Colors.white,
                          elevation: 0, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))),
                      child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                        const Icon(Icons.add_circle_rounded, size: 18),
                        const SizedBox(width: 8),
                        Text('Add to Complaints List',
                            style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w600)),
                      ]),
                    )),
                ]),
              ),
            ]),
          ),
        ],
      ]),
    );
  }

  // ── Helpers ────────────────────────────────────────────────────────
  Widget _resultBox(String label, String value, Color color) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 9),
    decoration: BoxDecoration(color: color.withOpacity(0.06),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withOpacity(0.2), width: 0.5)),
    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(label, style: GoogleFonts.dmSans(fontSize: 10, color: AppTheme.textMuted)),
      const SizedBox(height: 2),
      Text(value, style: GoogleFonts.dmSans(fontSize: 12, fontWeight: FontWeight.w700, color: color)),
    ]),
  );

  // ── Detail bottom sheet ───────────────────────────────────────────
  void _showDetail(Map<String, dynamic> c) {
    String status = c['status'] as String;
    showModalBottomSheet(context: context, isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => StatefulBuilder(builder: (ctx, set) => Container(
        height: MediaQuery.of(context).size.height * 0.82,
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
        decoration: const BoxDecoration(color: AppTheme.bg,
            borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
        child: Column(children: [
          Center(child: Container(width: 40, height: 4, margin: const EdgeInsets.only(bottom: 14),
              decoration: BoxDecoration(color: AppTheme.border, borderRadius: BorderRadius.circular(2)))),
          Row(children: [
            Expanded(child: Text('Complaint ${c['id']}',
                style: GoogleFonts.playfairDisplay(fontSize: 17, fontWeight: FontWeight.w700, color: AppTheme.textPrimary))),
            IconButton(icon: const Icon(Icons.close_rounded), onPressed: () => Navigator.pop(ctx)),
          ]),
          Expanded(child: SingleChildScrollView(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const SizedBox(height: 8),
            _dRow(Icons.person_rounded,        'Student',    c['student'] as String),
            _dRow(Icons.meeting_room_rounded,  'Room',       'Room ${c['room']}'),
            _dRow(Icons.category_rounded,      'Category',   c['category'] as String),
            if (c['subcategory'] != null)
              _dRow(Icons.subdirectory_arrow_right_rounded, 'Subcategory', c['subcategory'] as String,
                  color: const Color(0xFF185FA5)),
            _dRow(Icons.flag_rounded,          'Priority',   c['priority'] as String,
                color: _pColor(c['priority'] as String)),
            if (c['confidence'] != null)
              _dRow(Icons.psychology_rounded,  'AI Confidence',
                  '${((c['confidence'] as double) * 100).round()}%',
                  color: const Color(0xFF185FA5)),
            const SizedBox(height: 10),
            // Complaint text
            Container(padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: AppTheme.card, borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: AppTheme.border, width: 0.5)),
              child: Text(c['text'] as String,
                  style: GoogleFonts.dmSans(fontSize: 13, color: AppTheme.textSecondary, height: 1.5))),
            const SizedBox(height: 12),
            // Suggestion
            Container(padding: const EdgeInsets.all(13),
              decoration: BoxDecoration(color: const Color(0xFFE3EEF9),
                  borderRadius: BorderRadius.circular(12),
                  border: const Border(left: BorderSide(color: Color(0xFF185FA5), width: 3))),
              child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Icon(Icons.lightbulb_rounded, color: Color(0xFF185FA5), size: 16),
                const SizedBox(width: 10),
                Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('AI Suggested Action', style: GoogleFonts.dmSans(
                      fontSize: 11, fontWeight: FontWeight.w700, color: const Color(0xFF0D4F8C))),
                  const SizedBox(height: 4),
                  Text(c['suggestion'] as String, style: GoogleFonts.dmSans(
                      fontSize: 12, color: const Color(0xFF185FA5), height: 1.4)),
                ])),
              ])),
            const SizedBox(height: 14),
            // Status update
            Text('Update Status', style: GoogleFonts.dmSans(
                fontSize: 13, fontWeight: FontWeight.w600, color: AppTheme.textPrimary)),
            const SizedBox(height: 8),
            Row(children: ['Pending', 'In Progress', 'Resolved'].map((s) => Expanded(child: Padding(
              padding: EdgeInsets.only(right: s != 'Resolved' ? 8 : 0),
              child: GestureDetector(
                onTap: () { set(() => status = s); setState(() => c['status'] = s); },
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 180),
                  padding: const EdgeInsets.symmetric(vertical: 10),
                  decoration: BoxDecoration(
                    color: status == s ? _sColor(s) : AppTheme.card,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: status == s ? _sColor(s) : AppTheme.border, width: 0.5)),
                  child: Text(s, textAlign: TextAlign.center,
                      style: GoogleFonts.dmSans(fontSize: 11, fontWeight: FontWeight.w600,
                          color: status == s ? Colors.white : AppTheme.textSecondary)),
                ))))).toList()),
            const SizedBox(height: 14),
            SizedBox(width: double.infinity, height: 48,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.pop(ctx);
                  ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                    content: Text('Status updated to $status', style: GoogleFonts.dmSans()),
                    backgroundColor: AppTheme.teal, behavior: SnackBarBehavior.floating,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    margin: const EdgeInsets.all(16)));
                },
                style: ElevatedButton.styleFrom(backgroundColor: AppTheme.teal,
                    foregroundColor: Colors.white, elevation: 0,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))),
                child: Text('Save Update',
                    style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w600)),
              )),
          ]))),
        ]),
      )),
    );
  }

  Widget _dRow(IconData icon, String l, String v, {Color? color}) => Padding(
    padding: const EdgeInsets.only(bottom: 8),
    child: Row(children: [
      Icon(icon, size: 14, color: AppTheme.textMuted), const SizedBox(width: 8),
      Text('$l:', style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textMuted)),
      const SizedBox(width: 8),
      Expanded(child: Text(v, style: GoogleFonts.dmSans(
          fontSize: 12, fontWeight: FontWeight.w600, color: color ?? AppTheme.textPrimary))),
    ]),
  );
}


// ══════════════════════════════════════════════════════════════════
// 2. ATTENDANCE SCREEN
// ══════════════════════════════════════════════════════════════════
class WardenAttendanceScreen extends StatefulWidget {
  final String hostelName;
  const WardenAttendanceScreen({super.key, required this.hostelName});
  @override State<WardenAttendanceScreen> createState() => _WardenAttendanceState();
}
class _WardenAttendanceState extends State<WardenAttendanceScreen> {
  DateTime _date = DateTime.now();
  final _students = [
    {'name':'Sara Khan',  'room':'102','status':'IN', 'time':'9:32 PM','phone':'0300-1234567'},
    {'name':'Amna Raza',  'room':'201','status':'OUT','time':'11:00 AM','phone':'0311-1234567'},
    {'name':'Hina Malik', 'room':'301','status':'IN', 'time':'8:45 PM','phone':'0321-1234567'},
    {'name':'Fatima Ali', 'room':'103','status':'IN', 'time':'10:05 PM','phone':'0345-1234567'},
    {'name':'Zara Ahmed', 'room':'202','status':'OUT','time':'9:00 AM','phone':'0333-1234567'},
    {'name':'Maha Shah',  'room':'303','status':'IN', 'time':'7:50 PM','phone':'0356-1234567'},
  ];
  @override
  Widget build(BuildContext context) {
    final inC = _students.where((s)=>s['status']=='IN').length;
    return ResponsiveShell(child:Scaffold(
      backgroundColor:AppTheme.bg,
      appBar:_appBar(context,'Attendance'),
      body:Column(children:[
        Container(color:AppTheme.tealDeep,padding:const EdgeInsets.symmetric(vertical:10,horizontal:24),
          child:Row(mainAxisAlignment:MainAxisAlignment.spaceAround,children:[
            _sc('Total',_students.length,Colors.white),
            _sc('IN',inC,AppTheme.tealMint),
            _sc('OUT',_students.length-inC,Colors.orange.shade300),
          ])),
        Padding(padding:const EdgeInsets.fromLTRB(16,12,16,8),
          child:GestureDetector(
            onTap:()async{final d=await showDatePicker(context:context,initialDate:_date,
              firstDate:DateTime.now().subtract(const Duration(days:30)),lastDate:DateTime.now(),
              builder:(ctx,child)=>Theme(data:Theme.of(ctx).copyWith(
                  colorScheme:const ColorScheme.light(primary:AppTheme.teal,onPrimary:Colors.white)),child:child!));
              if(d!=null)setState(()=>_date=d);},
            child:Container(padding:const EdgeInsets.symmetric(horizontal:16,vertical:11),
              decoration:BoxDecoration(color:AppTheme.card,borderRadius:BorderRadius.circular(12),
                  border:Border.all(color:AppTheme.teal.withOpacity(0.4))),
              child:Row(children:[
                const Icon(Icons.calendar_today_rounded,color:AppTheme.teal,size:16),
                const SizedBox(width:10),
                Text('${_date.day}/${_date.month}/${_date.year}',
                    style:GoogleFonts.dmSans(fontSize:14,fontWeight:FontWeight.w600,color:AppTheme.textPrimary)),
                const Spacer(),
                const Icon(Icons.edit_calendar_rounded,color:AppTheme.textMuted,size:14),
              ])),
          )),
        Expanded(child:ListView.builder(
          padding:const EdgeInsets.fromLTRB(16,4,16,20),
          itemCount:_students.length,
          itemBuilder:(_,i){
            final s=_students[i];final isIn=s['status']=='IN';
            return Container(margin:const EdgeInsets.only(bottom:8),padding:const EdgeInsets.all(13),
              decoration:BoxDecoration(color:AppTheme.card,borderRadius:BorderRadius.circular(14),
                  border:Border.all(color:AppTheme.border,width:0.5)),
              child:Row(children:[
                Container(width:8,height:8,decoration:BoxDecoration(shape:BoxShape.circle,
                    color:isIn?AppTheme.tealLight:Colors.orange.shade400)),
                const SizedBox(width:12),
                Expanded(child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
                  Text(s['name']!,style:GoogleFonts.dmSans(fontSize:13,fontWeight:FontWeight.w600,color:AppTheme.textPrimary)),
                  Text('Room ${s['room']} · ${s['phone']}',style:GoogleFonts.dmSans(fontSize:11,color:AppTheme.textMuted)),
                ])),
                Container(padding:const EdgeInsets.symmetric(horizontal:10,vertical:4),
                  decoration:BoxDecoration(color:isIn?AppTheme.greenLight:Colors.orange.shade50,borderRadius:BorderRadius.circular(8)),
                  child:Text('${s['status']} ${s['time']}',style:GoogleFonts.dmSans(
                      fontSize:10,fontWeight:FontWeight.w600,
                      color:isIn?AppTheme.green:Colors.orange.shade700))),
              ]));
          })),
      ]),
    ));
  }
  Widget _sc(String l,int n,Color c)=>Column(children:[
    Text('$n',style:GoogleFonts.dmSans(color:c,fontSize:18,fontWeight:FontWeight.w700)),
    Text(l,style:GoogleFonts.dmSans(color:Colors.white60,fontSize:11)),
  ]);
}


// ══════════════════════════════════════════════════════════════════
// 3. ANNOUNCEMENTS SCREEN
// ══════════════════════════════════════════════════════════════════
class WardenAnnouncementsScreen extends StatefulWidget {
  final String hostelName;
  const WardenAnnouncementsScreen({super.key, required this.hostelName});
  @override State<WardenAnnouncementsScreen> createState() => _WardenAnnouncementsState();
}
class _WardenAnnouncementsState extends State<WardenAnnouncementsScreen> {
  final _titleCtrl = TextEditingController();
  final _bodyCtrl  = TextEditingController();
  String _audience = 'All Students';
  final _announcements = <Map<String,dynamic>>[
    {'title':'Mess Timing Update','body':'Dinner now served 7–9 PM starting Monday.','audience':'All Students','time':'Today 2:00 PM','color':AppTheme.teal},
    {'title':'Cleaning Schedule','body':'Deep cleaning Sunday 10 AM. Please cooperate.','audience':'All Students','time':'Yesterday','color':const Color(0xFF2D6A11)},
    {'title':'Fee Reminder','body':'Monthly fee due by 15th. Submit to avoid fine.','audience':'All Students','time':'2 days ago','color':const Color(0xFFB85C00)},
  ];

  void _post(){
    if(_titleCtrl.text.trim().isEmpty||_bodyCtrl.text.trim().isEmpty)return;
    setState((){
      _announcements.insert(0,{'title':_titleCtrl.text.trim(),'body':_bodyCtrl.text.trim(),
        'audience':_audience,'time':'Just now','color':AppTheme.teal});
      _titleCtrl.clear();_bodyCtrl.clear();
    });
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) => ResponsiveShell(child:Scaffold(
    backgroundColor:AppTheme.bg,
    appBar:_appBar(context,'Announcements',actions:[
      Padding(padding:const EdgeInsets.only(right:12),
        child:TextButton.icon(onPressed:()=>_showSheet(),
          icon:const Icon(Icons.add_rounded,size:18,color:Colors.white),
          label:Text('Post',style:GoogleFonts.dmSans(color:Colors.white,fontSize:13))))]),
    body:ListView.builder(
      padding:const EdgeInsets.fromLTRB(16,14,16,20),
      itemCount:_announcements.length,
      itemBuilder:(_,i){final a=_announcements[i];final color=a['color'] as Color;
        return Container(margin:const EdgeInsets.only(bottom:10),padding:const EdgeInsets.all(14),
          decoration:BoxDecoration(color:AppTheme.card,borderRadius:BorderRadius.circular(16),
              border:Border(left:BorderSide(color:color,width:4),
                  top:BorderSide(color:AppTheme.border,width:0.5),
                  right:BorderSide(color:AppTheme.border,width:0.5),
                  bottom:BorderSide(color:AppTheme.border,width:0.5))),
          child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
            Row(children:[
              Expanded(child:Text(a['title'] as String,style:GoogleFonts.dmSans(
                  fontSize:14,fontWeight:FontWeight.w700,color:AppTheme.textPrimary))),
              Container(padding:const EdgeInsets.symmetric(horizontal:8,vertical:3),
                decoration:BoxDecoration(color:color.withOpacity(0.1),borderRadius:BorderRadius.circular(6)),
                child:Text(a['audience'] as String,style:GoogleFonts.dmSans(
                    fontSize:10,fontWeight:FontWeight.w600,color:color))),
            ]),
            const SizedBox(height:6),
            Text(a['body'] as String,style:GoogleFonts.dmSans(
                fontSize:12,color:AppTheme.textSecondary,height:1.4)),
            const SizedBox(height:6),
            Row(children:[Icon(Icons.access_time_rounded,size:11,color:AppTheme.textMuted),
              const SizedBox(width:4),Text(a['time'] as String,
                  style:GoogleFonts.dmSans(fontSize:10,color:AppTheme.textMuted))]),
          ]));}),
  ));

  void _showSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => StatefulBuilder(
        builder: (ctx, set) => Padding(
          padding: EdgeInsets.only(bottom: MediaQuery.of(ctx).viewInsets.bottom),
          child: Container(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
            decoration: const BoxDecoration(
              color: AppTheme.bg,
              borderRadius: BorderRadius.vertical(top: Radius.circular(28))
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Center(
                  child: Container(
                    width: 40,
                    height: 4,
                    margin: const EdgeInsets.only(bottom: 14),
                    decoration: BoxDecoration(
                      color: AppTheme.border,
                      borderRadius: BorderRadius.circular(2)
                    )
                  )
                ),
                Row(
                  children: [
                    const Icon(Icons.campaign_rounded, color: AppTheme.teal, size: 20),
                    const SizedBox(width: 8),
                    Text('New Announcement', style: GoogleFonts.playfairDisplay(
                      fontSize: 17,
                      fontWeight: FontWeight.w700,
                      color: AppTheme.textPrimary
                    ))
                  ]
                ),
                const SizedBox(height: 14),
                _field(_titleCtrl, 'Title', Icons.title_rounded),
                const SizedBox(height: 10),
                _field(_bodyCtrl, 'Message...', Icons.notes_rounded, lines: 3),
                const SizedBox(height: 10),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: ['All Students', 'Floor 1', 'Floor 2', 'Floor 3', 'Parents'].map((a) =>
                    GestureDetector(
                      onTap: () => set(() => _audience = a),
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 160),
                        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                        decoration: BoxDecoration(
                          color: _audience == a ? AppTheme.teal : AppTheme.card,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(
                            color: _audience == a ? AppTheme.teal : AppTheme.border,
                            width: 0.5
                          )
                        ),
                        child: Text(
                          a,
                          style: GoogleFonts.dmSans(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: _audience == a ? Colors.white : AppTheme.textPrimary
                          )
                        )
                      )
                    )
                  ).toList()
                ),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  height: 48,
                  child: ElevatedButton(
                    onPressed: _post,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.teal,
                      foregroundColor: Colors.white,
                      elevation: 0,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))
                    ),
                    child: Text('Post Announcement', style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w600))
                  )
                )
              ]
            )
          )
        )
      )
    );
  }
}


// ══════════════════════════════════════════════════════════════════
// 4. ASSIGN PENALTIES SCREEN
// ══════════════════════════════════════════════════════════════════
class WardenPenaltiesScreen extends StatefulWidget {
  final String hostelName;
  const WardenPenaltiesScreen({super.key, required this.hostelName});
  @override State<WardenPenaltiesScreen> createState() => _WardenPenaltiesState();
}
class _WardenPenaltiesState extends State<WardenPenaltiesScreen> {
  final _penalties = <Map<String,dynamic>>[
    {'student':'Sara Khan','room':'102','reason':'Late return after curfew','amount':'PKR 500','date':'12 May 2026','status':'Paid'},
    {'student':'Amna Raza','room':'201','reason':'Unauthorized guest in room','amount':'PKR 1,000','date':'10 May 2026','status':'Unpaid'},
    {'student':'Zara Ahmed','room':'202','reason':'Noise violation after 11 PM','amount':'PKR 500','date':'8 May 2026','status':'Paid'},
  ];
  final _studentCtrl=TextEditingController(),_reasonCtrl=TextEditingController(),_amountCtrl=TextEditingController();

  void _addPenalty(){
    if(_studentCtrl.text.trim().isEmpty||_reasonCtrl.text.trim().isEmpty)return;
    setState((){
      _penalties.insert(0,{'student':_studentCtrl.text.trim(),'room':'?',
        'reason':_reasonCtrl.text.trim(),'amount':'PKR ${_amountCtrl.text.trim()}',
        'date':'Today','status':'Unpaid'});
      _studentCtrl.clear();_reasonCtrl.clear();_amountCtrl.clear();
    });
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context)=>ResponsiveShell(child:Scaffold(
    backgroundColor:AppTheme.bg,
    appBar:_appBar(context,'Assign Penalties',actions:[
      Padding(padding:const EdgeInsets.only(right:12),
        child:TextButton.icon(onPressed:_showSheet,
          icon:const Icon(Icons.add_rounded,size:18,color:Colors.white),
          label:Text('Add',style:GoogleFonts.dmSans(color:Colors.white,fontSize:13))))]),
    body:ListView.builder(
      padding:const EdgeInsets.fromLTRB(16,14,16,20),
      itemCount:_penalties.length,
      itemBuilder:(_,i){final p=_penalties[i];final paid=p['status']=='Paid';
        return Container(margin:const EdgeInsets.only(bottom:10),padding:const EdgeInsets.all(14),
          decoration:BoxDecoration(color:AppTheme.card,borderRadius:BorderRadius.circular(16),
              border:Border.all(color:AppTheme.border,width:0.5)),
          child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
            Row(children:[
              Expanded(child:Text(p['student'] as String,style:GoogleFonts.dmSans(
                  fontSize:14,fontWeight:FontWeight.w700,color:AppTheme.textPrimary))),
              _badge(p['status'] as String,paid?AppTheme.greenLight:Colors.red.shade50,
                  paid?AppTheme.green:Colors.red.shade700),
            ]),
            const SizedBox(height:5),
            Row(children:[_badge('Room ${p['room']}',AppTheme.bgSecondary,AppTheme.textSecondary),
              const SizedBox(width:6),_badge(p['date'] as String,AppTheme.bgSecondary,AppTheme.textMuted)]),
            const SizedBox(height:8),
            Text(p['reason'] as String,style:GoogleFonts.dmSans(fontSize:12,color:AppTheme.textSecondary)),
            const SizedBox(height:6),
            Row(children:[
              Text(p['amount'] as String,style:GoogleFonts.dmSans(fontSize:14,fontWeight:FontWeight.w700,color:Colors.red.shade700)),
              const Spacer(),
              if(!paid)GestureDetector(
                onTap:()=>setState(()=>_penalties[i]['status']='Paid'),
                child:Container(padding:const EdgeInsets.symmetric(horizontal:12,vertical:5),
                  decoration:BoxDecoration(color:AppTheme.teal,borderRadius:BorderRadius.circular(8)),
                  child:Text('Mark Paid',style:GoogleFonts.dmSans(fontSize:11,fontWeight:FontWeight.w600,color:Colors.white)))),
            ]),
          ]));
      }),
    floatingActionButton:FloatingActionButton.extended(onPressed:_showSheet,
      backgroundColor:AppTheme.teal,
      icon:const Icon(Icons.gavel_rounded,color:Colors.white),
      label:Text('Add Penalty',style:GoogleFonts.dmSans(color:Colors.white,fontWeight:FontWeight.w600))),
  ));

  void _showSheet(){
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => Padding(
        padding: EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
        child: Container(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
          decoration: const BoxDecoration(
            color: AppTheme.bg,
            borderRadius: BorderRadius.vertical(top: Radius.circular(28))
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  margin: const EdgeInsets.only(bottom: 14),
                  decoration: BoxDecoration(
                    color: AppTheme.border,
                    borderRadius: BorderRadius.circular(2)
                  )
                )
              ),
              Row(
                children: [
                  Icon(Icons.gavel_rounded, color: Colors.red.shade600, size: 20),
                  const SizedBox(width: 8),
                  Text('Assign Penalty', style: GoogleFonts.playfairDisplay(
                    fontSize: 17,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary
                  ))
                ]
              ),
              const SizedBox(height: 14),
              _field(_studentCtrl, 'Student Name *', Icons.person_rounded),
              const SizedBox(height: 10),
              _field(_reasonCtrl, 'Reason for penalty *', Icons.notes_rounded, lines: 2),
              const SizedBox(height: 10),
              _field(_amountCtrl, 'Amount (PKR)', Icons.payments_rounded, type: TextInputType.number),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                height: 48,
                child: ElevatedButton(
                  onPressed: _addPenalty,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red.shade600,
                    foregroundColor: Colors.white,
            elevation: 0,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))
                  ),
                  child: Text('Assign Penalty', style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w600))
                )
              )
            ]
          )
        )
      )
    );
  }
}


// ══════════════════════════════════════════════════════════════════
// 5. ROOM ALLOCATION SCREEN
// ══════════════════════════════════════════════════════════════════
class WardenRoomAllocationScreen extends StatefulWidget {
  final String hostelName;
  const WardenRoomAllocationScreen({super.key, required this.hostelName});
  @override State<WardenRoomAllocationScreen> createState() => _WardenRoomAllocationState();
}
class _WardenRoomAllocationState extends State<WardenRoomAllocationScreen> {
  final _rooms = [
    {'number':'101','type':'Single',   'capacity':1,'occupied':1,'students':['Sara Khan'],   'price':'15,000'},
    {'number':'102','type':'Double',   'capacity':2,'occupied':2,'students':['Amna Raza','Hina Malik'],'price':'10,000'},
    {'number':'103','type':'Double',   'capacity':2,'occupied':1,'students':['Fatima Ali'],  'price':'10,000'},
    {'number':'201','type':'Triple',   'capacity':3,'occupied':3,'students':['Zara Ahmed','Maha Shah','Aliya Iqbal'],'price':'8,000'},
    {'number':'202','type':'Single',   'capacity':1,'occupied':0,'students':[],              'price':'15,000'},
    {'number':'203','type':'Dormitory','capacity':6,'occupied':4,'students':['Sana Qureshi','Nida Ali','Rabia Khan','Iqra Baig'],'price':'5,000'},
    {'number':'301','type':'Triple',   'capacity':3,'occupied':0,'students':[],              'price':'8,000'},
  ];

  @override
  Widget build(BuildContext context){
    final total=_rooms.fold<int>(0,(s,r)=>s+(r['capacity'] as int));
    final occupied=_rooms.fold<int>(0,(s,r)=>s+(r['occupied'] as int));
    return ResponsiveShell(child:Scaffold(
      backgroundColor:AppTheme.bg,
      appBar:_appBar(context,'Room Allocation'),
      body:Column(children:[
        Container(color:AppTheme.tealDeep,padding:const EdgeInsets.symmetric(vertical:10,horizontal:24),
          child:Row(mainAxisAlignment:MainAxisAlignment.spaceAround,children:[
            _sc('Total Beds',total,Colors.white),
            _sc('Occupied',occupied,Colors.orange.shade300),
            _sc('Available',total-occupied,AppTheme.tealMint),
            _sc('Rooms',_rooms.length,Colors.white70),
          ])),
        Expanded(child:ListView.builder(
          padding:const EdgeInsets.fromLTRB(16,12,16,20),
          itemCount:_rooms.length,
          itemBuilder:(_,i){final r=_rooms[i];
            final avail=(r['capacity'] as int)-(r['occupied'] as int);
            final isFull=avail==0;
            return GestureDetector(
              onTap:()=>_showRoomDetail(r),
              child:Container(margin:const EdgeInsets.only(bottom:10),padding:const EdgeInsets.all(14),
                decoration:BoxDecoration(color:AppTheme.card,borderRadius:BorderRadius.circular(16),
                    border:Border.all(color:AppTheme.border,width:0.5)),
                child:Row(children:[
                  Container(width:50,height:50,decoration:BoxDecoration(
                      color:AppTheme.teal.withOpacity(0.1),borderRadius:BorderRadius.circular(12)),
                    child:Column(mainAxisAlignment:MainAxisAlignment.center,children:[
                      Text(r['number'] as String,style:GoogleFonts.dmSans(fontSize:14,fontWeight:FontWeight.w700,color:AppTheme.teal)),
                      Text(r['type'] as String,style:GoogleFonts.dmSans(fontSize:9,color:AppTheme.textMuted)),
                    ])),
                  const SizedBox(width:12),
                  Expanded(child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
                    Text('Room ${r['number']} — ${r['type']}',style:GoogleFonts.dmSans(
                        fontSize:13,fontWeight:FontWeight.w700,color:AppTheme.textPrimary)),
                    Text('PKR ${r['price']}/month · ${r['occupied']}/${r['capacity']} beds',
                        style:GoogleFonts.dmSans(fontSize:11,color:AppTheme.textMuted)),
                    const SizedBox(height:5),
                    Row(children:List.generate(r['capacity'] as int,(bi)=>Container(
                      width:10,height:10,margin:const EdgeInsets.only(right:4),
                      decoration:BoxDecoration(shape:BoxShape.circle,
                          color:bi<(r['occupied'] as int)?AppTheme.teal:AppTheme.bgSecondary,
                          border:Border.all(color:AppTheme.border,width:0.5))))),
                  ])),
                  Container(padding:const EdgeInsets.symmetric(horizontal:10,vertical:5),
                    decoration:BoxDecoration(color:isFull?Colors.red.shade50:AppTheme.greenLight,
                        borderRadius:BorderRadius.circular(8)),
                    child:Text(isFull?'Full':'$avail free',style:GoogleFonts.dmSans(
                        fontSize:11,fontWeight:FontWeight.w600,
                        color:isFull?Colors.red.shade700:AppTheme.green))),
                ])));
          })),
      ]),
    ));
  }

  void _showRoomDetail(Map<String,dynamic> r){
    showModalBottomSheet(context:context,isScrollControlled:true,backgroundColor:Colors.transparent,
      builder:(_)=>Container(
        height:MediaQuery.of(context).size.height*0.60,
        padding:const EdgeInsets.fromLTRB(20,16,20,28),
        decoration:const BoxDecoration(color:AppTheme.bg,borderRadius:BorderRadius.vertical(top:Radius.circular(28))),
        child:Column(children:[
          Center(child:Container(width:40,height:4,margin:const EdgeInsets.only(bottom:14),
              decoration:BoxDecoration(color:AppTheme.border,borderRadius:BorderRadius.circular(2)))),
          Text('Room ${r['number']} — ${r['type']}',style:GoogleFonts.playfairDisplay(
              fontSize:17,fontWeight:FontWeight.w700,color:AppTheme.textPrimary)),
          const SizedBox(height:14),
          Expanded(child:ListView(children:[
            if((r['students'] as List).isEmpty)
              Center(child:Column(children:[const SizedBox(height:20),
                Icon(Icons.bed_outlined,size:48,color:AppTheme.textMuted.withOpacity(0.3)),
                const SizedBox(height:10),Text('Room is vacant',style:GoogleFonts.dmSans(color:AppTheme.textMuted))]))
            else
              ...(r['students'] as List).map((s)=>Container(
                margin:const EdgeInsets.only(bottom:8),padding:const EdgeInsets.all(12),
                decoration:BoxDecoration(color:AppTheme.bg,borderRadius:BorderRadius.circular(12),
                    border:Border.all(color:AppTheme.border,width:0.5)),
                child:Row(children:[
                  Container(width:36,height:36,decoration:BoxDecoration(
                      color:AppTheme.teal.withOpacity(0.1),shape:BoxShape.circle),
                    child:const Icon(Icons.person_rounded,size:18,color:AppTheme.teal)),
                  const SizedBox(width:10),
                  Text(s as String,style:GoogleFonts.dmSans(fontSize:13,fontWeight:FontWeight.w600,color:AppTheme.textPrimary)),
                ]))),
            if((r['capacity'] as int)-(r['occupied'] as int)>0)...[
              const SizedBox(height:10),
              Container(padding:const EdgeInsets.all(12),
                decoration:BoxDecoration(color:AppTheme.greenLight,borderRadius:BorderRadius.circular(12),
                    border:Border.all(color:AppTheme.green.withOpacity(0.3))),
                child:Row(children:[Icon(Icons.bed_rounded,color:AppTheme.green,size:16),
                  const SizedBox(width:8),
                  Text('${(r['capacity'] as int)-(r['occupied'] as int)} bed(s) available for allocation',
                      style:GoogleFonts.dmSans(fontSize:12,color:AppTheme.green))])),
            ],
          ])),
        ]),
      ));
  }
  Widget _sc(String l,int n,Color c)=>Column(children:[
    Text('$n',style:GoogleFonts.dmSans(color:c,fontSize:18,fontWeight:FontWeight.w700)),
    Text(l,style:GoogleFonts.dmSans(color:Colors.white60,fontSize:11)),
  ]);
}


// ══════════════════════════════════════════════════════════════════
// 6. VIEW STUDENT & PARENT DETAILS
// ══════════════════════════════════════════════════════════════════
class WardenStudentDetailsScreen extends StatefulWidget {
  final String hostelName;
  const WardenStudentDetailsScreen({super.key, required this.hostelName});
  @override State<WardenStudentDetailsScreen> createState() => _WardenStudentDetailsState();
}
class _WardenStudentDetailsState extends State<WardenStudentDetailsScreen> {
  String _query = '';
  final _ctrl   = TextEditingController();
  final _students = [
    {'name':'Sara Khan',  'room':'102','uni':'FAST-NU','phone':'0300-1234567','fee':'Paid',  'status':'IN', 'cnic':'35202-1234567-8','semester':'5th','parent':'Ahmed Khan','parentPhone':'0321-9876543','parentRel':'Father'},
    {'name':'Amna Raza',  'room':'201','uni':'NUST',   'phone':'0311-1234567','fee':'Pending','status':'OUT','cnic':'35202-2345678-9','semester':'3rd','parent':'Raza Shah','parentPhone':'0333-8765432','parentRel':'Father'},
    {'name':'Hina Malik', 'room':'301','uni':'QAU',    'phone':'0321-1234567','fee':'Paid',  'status':'IN', 'cnic':'35202-3456789-0','semester':'7th','parent':'Malik Hussain','parentPhone':'0300-7654321','parentRel':'Father'},
    {'name':'Fatima Ali', 'room':'103','uni':'FAST-NU','phone':'0345-1234567','fee':'Overdue','status':'IN','cnic':'35202-4567890-1','semester':'2nd','parent':'Ali Hassan','parentPhone':'0311-6543210','parentRel':'Uncle'},
  ];
  List<Map<String,dynamic>> get _filtered=>_students.where((s)=>_query.isEmpty||
      s['name']!.toLowerCase().contains(_query.toLowerCase())||
      s['room']!.contains(_query)).toList();

  Color _feeColor(String f)=>f=='Paid'?AppTheme.green:f=='Pending'?Colors.orange.shade700:Colors.red.shade700;

  @override
  Widget build(BuildContext context)=>ResponsiveShell(child:Scaffold(
    backgroundColor:AppTheme.bg,
    appBar:_appBar(context,'Student & Parent Details'),
    body:Column(children:[
      Padding(padding:const EdgeInsets.fromLTRB(16,12,16,8),
        child:Container(decoration:BoxDecoration(color:AppTheme.card,borderRadius:BorderRadius.circular(12),
            border:Border.all(color:AppTheme.border,width:0.5)),
          child:TextField(controller:_ctrl,onChanged:(v)=>setState(()=>_query=v),
            style:GoogleFonts.dmSans(fontSize:14,color:AppTheme.textPrimary),
            decoration:InputDecoration(hintText:'Search by name or room',
              hintStyle:GoogleFonts.dmSans(color:AppTheme.textMuted,fontSize:13),
              prefixIcon:const Icon(Icons.search_rounded,size:18,color:AppTheme.textMuted),
              border:InputBorder.none,contentPadding:const EdgeInsets.symmetric(horizontal:16,vertical:13))))),
      Expanded(child:ListView.builder(
        padding:const EdgeInsets.fromLTRB(16,4,16,20),
        itemCount:_filtered.length,
        itemBuilder:(_,i){final s=_filtered[i];
          return GestureDetector(onTap:()=>_showProfile(s),
            child:Container(margin:const EdgeInsets.only(bottom:10),padding:const EdgeInsets.all(14),
              decoration:BoxDecoration(color:AppTheme.card,borderRadius:BorderRadius.circular(16),
                  border:Border.all(color:AppTheme.border,width:0.5)),
              child:Row(children:[
                Container(width:44,height:44,decoration:BoxDecoration(
                    color:AppTheme.teal.withOpacity(0.1),shape:BoxShape.circle),
                  child:const Icon(Icons.person_rounded,size:22,color:AppTheme.teal)),
                const SizedBox(width:12),
                Expanded(child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
                  Text(s['name']!,style:GoogleFonts.dmSans(fontSize:13,fontWeight:FontWeight.w700,color:AppTheme.textPrimary)),
                  Text('${s['uni']} · Room ${s['room']} · ${s['semester']} Sem',
                      style:GoogleFonts.dmSans(fontSize:11,color:AppTheme.textMuted)),
                  const SizedBox(height:4),
                  Row(children:[
                    _badge(s['status']!,s['status']=='IN'?AppTheme.greenLight:Colors.orange.shade50,
                        s['status']=='IN'?AppTheme.green:Colors.orange.shade700),
                    const SizedBox(width:6),
                    _badge(s['fee']!,_feeColor(s['fee']!).withOpacity(0.1),_feeColor(s['fee']!)),
                  ]),
                ])),
                const Icon(Icons.chevron_right_rounded,color:AppTheme.textMuted,size:18),
              ])));
        })),
    ]),
  ));

  void _showProfile(Map<String,dynamic> s){
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => Container(
        height: MediaQuery.of(context).size.height * 0.75,
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
        decoration: const BoxDecoration(
          color: AppTheme.bg,
          borderRadius: BorderRadius.vertical(top: Radius.circular(28))
        ),
        child: Column(
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                margin: const EdgeInsets.only(bottom: 14),
                decoration: BoxDecoration(
                  color: AppTheme.border,
                  borderRadius: BorderRadius.circular(2)
                )
              )
            ),
            Row(
              children: [
                Container(
                  width: 50,
                  height: 50,
                  decoration: BoxDecoration(
                    color: AppTheme.teal.withOpacity(0.1),
                    shape: BoxShape.circle
                  ),
                  child: const Icon(Icons.person_rounded, size: 26, color: AppTheme.teal)
                ),
                const SizedBox(width: 12),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(s['name']!, style: GoogleFonts.dmSans(fontSize: 16, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
                    Text('Room ${s['room']} · ${s['uni']}', style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textMuted)),
                  ]
                ),
              ]
            ),
            const SizedBox(height: 14),
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  children: [
                    _section('Student Information', [
                      _pr('Phone', s['phone']!),
                      _pr('CNIC', s['cnic']!),
                      _pr('Semester', s['semester']!),
                      _pr('Fee Status', s['fee']!),
                    ]),
                    const SizedBox(height: 12),
                    _section('Parent / Guardian', [
                      _pr('Name', s['parent']!),
                      _pr('Phone', s['parentPhone']!),
                      _pr('Relation', s['parentRel']!),
                    ]),
                  ]
                )
              )
            ),
          ]
        )
      )
    );
  }

  Widget _section(String title,List<Widget> rows)=>Container(
    padding:const EdgeInsets.all(14),
    decoration:BoxDecoration(color:AppTheme.card,borderRadius:BorderRadius.circular(14),
        border:Border.all(color:AppTheme.border,width:0.5)),
    child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
      Text(title,style:GoogleFonts.dmSans(fontSize:13,fontWeight:FontWeight.w700,color:AppTheme.textPrimary)),
      const SizedBox(height:10),
      ...rows,
    ]));

  Widget _pr(String l,String v)=>Padding(padding:const EdgeInsets.only(bottom:8),
    child:Row(children:[
      SizedBox(width:100,child:Text(l,style:GoogleFonts.dmSans(fontSize:12,color:AppTheme.textMuted))),
      Expanded(child:Text(v,style:GoogleFonts.dmSans(fontSize:12,fontWeight:FontWeight.w600,color:AppTheme.textPrimary))),
    ]));
}


// ══════════════════════════════════════════════════════════════════
// 7. EMPTY BED NOTIFICATIONS
// ══════════════════════════════════════════════════════════════════
class WardenEmptyBedsScreen extends StatelessWidget {
  final String hostelName;
  const WardenEmptyBedsScreen({super.key, required this.hostelName});
  @override
  Widget build(BuildContext context){
    final vacancies=[
      {'room':'202','type':'Single','beds':1,'floor':'Floor 2','since':'2 days ago','price':'15,000'},
      {'room':'301','type':'Triple','beds':3,'floor':'Floor 3','since':'1 week ago','price':'8,000'},
      {'room':'103','type':'Double','beds':1,'floor':'Floor 1','since':'Today','price':'10,000'},
    ];
    return ResponsiveShell(child:Scaffold(
      backgroundColor:AppTheme.bg,
      appBar:_appBar(context,'Empty Bed Notifications'),
      body:Column(children:[
        Container(color:AppTheme.tealDeep,padding:const EdgeInsets.symmetric(vertical:10,horizontal:24),
          child:Row(mainAxisAlignment:MainAxisAlignment.spaceAround,children:[
            Column(children:[Text('${vacancies.length}',style:GoogleFonts.dmSans(color:AppTheme.tealMint,fontSize:18,fontWeight:FontWeight.w700)),
              Text('Vacant Rooms',style:GoogleFonts.dmSans(color:Colors.white60,fontSize:11))]),
            Column(children:[Text('${vacancies.fold<int>(0,(s,v)=>s+(v['beds'] as int))}',
                style:GoogleFonts.dmSans(color:Colors.orange.shade300,fontSize:18,fontWeight:FontWeight.w700)),
              Text('Empty Beds',style:GoogleFonts.dmSans(color:Colors.white60,fontSize:11))]),
          ])),
        Container(margin:const EdgeInsets.fromLTRB(16,12,16,4),padding:const EdgeInsets.all(12),
          decoration:BoxDecoration(color:AppTheme.teal.withOpacity(0.06),borderRadius:BorderRadius.circular(12),
              border:Border.all(color:AppTheme.teal.withOpacity(0.2))),
          child:Row(children:[const Icon(Icons.notifications_active_rounded,color:AppTheme.teal,size:16),
            const SizedBox(width:10),
            Expanded(child:Text('Parents and students are notified automatically when beds become vacant.',
                style:GoogleFonts.dmSans(fontSize:12,color:AppTheme.teal,height:1.4)))])),
        Expanded(child:ListView.builder(
          padding:const EdgeInsets.fromLTRB(16,8,16,20),
          itemCount:vacancies.length,
          itemBuilder:(_,i){final v=vacancies[i];
            return Container(margin:const EdgeInsets.only(bottom:10),padding:const EdgeInsets.all(14),
              decoration:BoxDecoration(color:AppTheme.card,borderRadius:BorderRadius.circular(16),
                  border:Border.all(color:AppTheme.green.withOpacity(0.3))),
              child:Row(children:[
                Container(width:50,height:50,decoration:BoxDecoration(
                    color:AppTheme.green.withOpacity(0.1),borderRadius:BorderRadius.circular(12)),
                  child:Column(mainAxisAlignment:MainAxisAlignment.center,children:[
                    Text(v['room'] as String,style:GoogleFonts.dmSans(fontSize:14,fontWeight:FontWeight.w700,color:AppTheme.green)),
                    Text(v['floor'] as String,style:GoogleFonts.dmSans(fontSize:9,color:AppTheme.textMuted)),
                  ])),
                const SizedBox(width:12),
                Expanded(child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
                  Text('${v['type']} Room — ${v['beds']} bed${(v['beds'] as int)>1?'s':''} free',
                      style:GoogleFonts.dmSans(fontSize:13,fontWeight:FontWeight.w700,color:AppTheme.textPrimary)),
                  Text('PKR ${v['price']}/month · Vacant since ${v['since']}',
                      style:GoogleFonts.dmSans(fontSize:11,color:AppTheme.textMuted)),
                ])),
                Container(padding:const EdgeInsets.symmetric(horizontal:10,vertical:5),
                  decoration:BoxDecoration(color:AppTheme.greenLight,borderRadius:BorderRadius.circular(8)),
                  child:Text('${v['beds']} free',style:GoogleFonts.dmSans(
                      fontSize:11,fontWeight:FontWeight.w700,color:AppTheme.green))),
              ]));
          })),
      ]),
    ));
  }
}


// ══════════════════════════════════════════════════════════════════
// 8. CHAT SCREEN (Parents & Students)
// ══════════════════════════════════════════════════════════════════
class WardenChatScreen extends StatefulWidget {
  final String hostelName;
  const WardenChatScreen({super.key, required this.hostelName});
  @override State<WardenChatScreen> createState() => _WardenChatState();
}
class _WardenChatState extends State<WardenChatScreen> {
  final _contacts = [
    {'name':'Sara Khan',  'role':'Student','room':'102','last':'Ok, thank you warden','time':'2m','unread':0,'online':true},
    {'name':'Ahmed Khan', 'role':'Parent', 'room':'102','last':'Is Sara doing well?','time':'15m','unread':2,'online':false},
    {'name':'Amna Raza',  'role':'Student','room':'201','last':'Can I come late tonight?','time':'1h','unread':1,'online':true},
    {'name':'Raza Shah',  'role':'Parent', 'room':'201','last':'Please let me know her attendance','time':'3h','unread':0,'online':false},
    {'name':'Hina Malik', 'role':'Student','room':'301','last':'Thank you for resolving my complaint','time':'1d','unread':0,'online':false},
  ];
  String _tab = 'All';

  List<Map<String,dynamic>> get _filtered=>_tab=='All'?_contacts:
      _contacts.where((c)=>c['role']==_tab).toList();

  @override
  Widget build(BuildContext context)=>ResponsiveShell(child:Scaffold(
    backgroundColor:AppTheme.bg,
    appBar:_appBar(context,'Chat'),
    body:Column(children:[
      Padding(padding:const EdgeInsets.fromLTRB(16,12,16,8),
        child:Row(children:['All','Student','Parent'].map((t)=>Padding(
          padding:EdgeInsets.only(right:t!='Parent'?8:0),
          child:GestureDetector(onTap:()=>setState(()=>_tab=t),
            child:AnimatedContainer(duration:const Duration(milliseconds:160),
              padding:const EdgeInsets.symmetric(horizontal:16,vertical:8),
              decoration:BoxDecoration(color:_tab==t?AppTheme.teal:AppTheme.card,
                  borderRadius:BorderRadius.circular(20),
                  border:Border.all(color:_tab==t?AppTheme.teal:AppTheme.border,width:0.5)),
              child:Text(t,style:GoogleFonts.dmSans(fontSize:13,fontWeight:FontWeight.w600,
                  color:_tab==t?Colors.white:AppTheme.textPrimary)))))).toList())),
      Expanded(child:ListView.builder(
        padding:const EdgeInsets.fromLTRB(16,4,16,20),
        itemCount:_filtered.length,
        itemBuilder:(_,i){final c=_filtered[i];
          return GestureDetector(onTap:()=>_openChat(c),
            child:Container(margin:const EdgeInsets.only(bottom:8),padding:const EdgeInsets.all(13),
              decoration:BoxDecoration(color:AppTheme.card,borderRadius:BorderRadius.circular(14),
                  border:Border.all(color:AppTheme.border,width:0.5)),
              child:Row(children:[
                Stack(children:[
                  Container(width:44,height:44,decoration:BoxDecoration(
                      color:c['role']=='Student'?AppTheme.teal.withOpacity(0.1):const Color(0xFF7B3FC4).withOpacity(0.1),
                      shape:BoxShape.circle),
                    child:Icon(c['role']=='Student'?Icons.school_rounded:Icons.family_restroom_rounded,
                        size:20,color:c['role']=='Student'?AppTheme.teal:const Color(0xFF7B3FC4))),
                  if(c['online'] as bool)Positioned(bottom:2,right:2,child:Container(width:10,height:10,
                      decoration:BoxDecoration(color:AppTheme.green,shape:BoxShape.circle,
                          border:Border.all(color:AppTheme.card,width:2)))),
                ]),
                const SizedBox(width:12),
                Expanded(child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
                  Row(children:[
                    Expanded(child:Text(c['name'] as String,style:GoogleFonts.dmSans(
                        fontSize:13,fontWeight:FontWeight.w700,color:AppTheme.textPrimary))),
                    Text(c['time'] as String,style:GoogleFonts.dmSans(fontSize:11,color:AppTheme.textMuted)),
                  ]),
                  Row(children:[
                    _badge(c['role'] as String,
                        c['role']=='Student'?AppTheme.teal.withOpacity(0.08):const Color(0xFF7B3FC4).withOpacity(0.08),
                        c['role']=='Student'?AppTheme.teal:const Color(0xFF7B3FC4)),
                    const SizedBox(width:6),
                    _badge('Room ${c['room']}',AppTheme.bgSecondary,AppTheme.textMuted),
                  ]),
                  const SizedBox(height:3),
                  Text(c['last'] as String,maxLines:1,overflow:TextOverflow.ellipsis,
                      style:GoogleFonts.dmSans(fontSize:11,color:AppTheme.textMuted)),
                ])),
                if((c['unread'] as int)>0)Container(width:20,height:20,
                  decoration:const BoxDecoration(color:AppTheme.teal,shape:BoxShape.circle),
                  child:Center(child:Text('${c['unread']}',style:GoogleFonts.dmSans(
                      fontSize:10,fontWeight:FontWeight.w700,color:Colors.white)))),
              ])));
        })),
    ]),
  ));

  void _openChat(Map<String,dynamic> contact) {
    final msgs = <Map<String,dynamic>>[
      {'from': 'them', 'text': 'Hello warden, I have a question.', 'time': '10:00 AM'},
      {'from': 'me', 'text': 'Sure, how can I help?', 'time': '10:01 AM'},
      {'from': 'them', 'text': contact['last'] as String, 'time': '10:05 AM'},
    ];
    final ctrl = TextEditingController();
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ResponsiveShell(
          child: Scaffold(
            backgroundColor: AppTheme.bg,
            appBar: AppBar(
              leading: IconButton(
                icon: const Icon(Icons.arrow_back_rounded),
                onPressed: () => Navigator.pop(context),
              ),
              title: Row(
                children: [
                  Container(
                    width: 34,
                    height: 34,
                    decoration: BoxDecoration(
                      color: AppTheme.teal.withOpacity(0.15),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(
                      contact['role'] == 'Student' ? Icons.school_rounded : Icons.family_restroom_rounded,
                      size: 16,
                      color: AppTheme.teal,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        contact['name'] as String,
                        style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w700, color: Colors.white),
                      ),
                      Text(
                        '${contact['role']} · Room ${contact['room']}',
                        style: GoogleFonts.dmSans(fontSize: 10, color: Colors.white60),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            body: StatefulBuilder(
              builder: (ctx, set) => Column(
                children: [
                  Expanded(
                    child: ListView.builder(
                      padding: const EdgeInsets.all(16),
                      itemCount: msgs.length,
                      itemBuilder: (_, i) {
                        final m = msgs[i];
                        final isMe = m['from'] == 'me';
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 10),
                          child: Row(
                            mainAxisAlignment: isMe ? MainAxisAlignment.end : MainAxisAlignment.start,
                            children: [
                              Container(
                                constraints: BoxConstraints(maxWidth: MediaQuery.of(ctx).size.width * 0.7),
                                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                                decoration: BoxDecoration(
                                  color: isMe ? AppTheme.teal : AppTheme.card,
                                  borderRadius: BorderRadius.only(
                                    topLeft: const Radius.circular(16),
                                    topRight: const Radius.circular(16),
                                    bottomLeft: Radius.circular(isMe ? 16 : 4),
                                    bottomRight: Radius.circular(isMe ? 4 : 16),
                                  ),
                                ),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.end,
                                  children: [
                                    Text(
                                      m['text'] as String,
                                      style: GoogleFonts.dmSans(
                                        fontSize: 13,
                                        color: isMe ? Colors.white : AppTheme.textPrimary,
                                      ),
                                    ),
                                    Text(
                                      m['time'] as String,
                                      style: GoogleFonts.dmSans(
                                        fontSize: 10,
                                        color: isMe ? Colors.white60 : AppTheme.textMuted,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
                  ),
                  Container(
                    padding: EdgeInsets.fromLTRB(12, 10, 12, 12 + MediaQuery.of(ctx).padding.bottom),
                    decoration: BoxDecoration(
                      color: AppTheme.card,
                      border: Border(top: BorderSide(color: AppTheme.border, width: 0.5)),
                    ),
                    child: Row(
                      children: [
                        Expanded(
                          child: Container(
                            decoration: BoxDecoration(
                              color: AppTheme.bg,
                              borderRadius: BorderRadius.circular(24),
                              border: Border.all(color: AppTheme.border, width: 0.5),
                            ),
                            child: TextField(
                              controller: ctrl,
                              style: GoogleFonts.dmSans(fontSize: 14, color: AppTheme.textPrimary),
                              decoration: InputDecoration(
                                hintText: 'Type a message...',
                                hintStyle: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 13),
                                border: InputBorder.none,
                                contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        GestureDetector(
                          onTap: () {
                            if (ctrl.text.trim().isEmpty) return;
                            set(() {
                              msgs.add({'from': 'me', 'text': ctrl.text.trim(), 'time': 'Now'});
                            });
                            ctrl.clear();
                          },
                          child: Container(
                            width: 44,
                            height: 44,
                            decoration: const BoxDecoration(color: AppTheme.teal, shape: BoxShape.circle),
                            child: const Icon(Icons.send_rounded, color: Colors.white, size: 18),
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
      ),
    );
  }
}


// ══════════════════════════════════════════════════════════════════
// 9. CLEANING SCHEDULE SCREEN
// ══════════════════════════════════════════════════════════════════
class WardenCleaningScheduleScreen extends StatefulWidget {
  final String hostelName;
  const WardenCleaningScheduleScreen({super.key, required this.hostelName});
  @override State<WardenCleaningScheduleScreen> createState() => _WardenCleaningState();
}
class _WardenCleaningState extends State<WardenCleaningScheduleScreen> {
  final _schedule = <Map<String,dynamic>>[
    {'area':'Room Floors (All)','day':'Monday & Thursday','time':'8:00 AM – 10:00 AM','staff':'Cleaning Team A','status':'Scheduled'},
    {'area':'Bathrooms (Floor 1)','day':'Daily','time':'6:00 AM & 6:00 PM','staff':'Cleaning Team B','status':'Scheduled'},
    {'area':'Bathrooms (Floor 2)','day':'Daily','time':'7:00 AM & 7:00 PM','staff':'Cleaning Team B','status':'Scheduled'},
    {'area':'Common Areas','day':'Tuesday & Friday','time':'9:00 AM – 11:00 AM','staff':'Cleaning Team A','status':'Scheduled'},
    {'area':'Kitchen & Mess','day':'Daily','time':'After every meal','staff':'Kitchen Staff','status':'Scheduled'},
    {'area':'Corridors','day':'Wednesday & Saturday','time':'8:00 AM – 9:00 AM','staff':'Cleaning Team C','status':'Scheduled'},
  ];
  final _areaCtrl=TextEditingController(),_dayCtrl=TextEditingController(),_timeCtrl=TextEditingController();

  void _add(){
    if(_areaCtrl.text.trim().isEmpty)return;
    setState((){_schedule.add({'area':_areaCtrl.text.trim(),'day':_dayCtrl.text.trim(),
      'time':_timeCtrl.text.trim(),'staff':'Unassigned','status':'Scheduled'});
      _areaCtrl.clear();_dayCtrl.clear();_timeCtrl.clear();
    });
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context)=>ResponsiveShell(child:Scaffold(
    backgroundColor:AppTheme.bg,
    appBar:_appBar(context,'Cleaning Schedule',actions:[
      Padding(padding:const EdgeInsets.only(right:12),
        child:TextButton.icon(onPressed:_showSheet,
          icon:const Icon(Icons.add_rounded,size:18,color:Colors.white),
          label:Text('Add',style:GoogleFonts.dmSans(color:Colors.white,fontSize:13))))]),
    body:ListView.builder(
      padding:const EdgeInsets.fromLTRB(16,14,16,20),
      itemCount:_schedule.length,
      itemBuilder:(_,i){final s=_schedule[i];
        return Container(margin:const EdgeInsets.only(bottom:10),padding:const EdgeInsets.all(14),
          decoration:BoxDecoration(color:AppTheme.card,borderRadius:BorderRadius.circular(16),
              border:Border.all(color:AppTheme.border,width:0.5)),
          child:Row(children:[
            Container(width:44,height:44,decoration:BoxDecoration(
                color:AppTheme.teal.withOpacity(0.1),borderRadius:BorderRadius.circular(12)),
              child:const Icon(Icons.cleaning_services_rounded,size:22,color:AppTheme.teal)),
            const SizedBox(width:12),
            Expanded(child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
              Text(s['area'] as String,style:GoogleFonts.dmSans(fontSize:13,fontWeight:FontWeight.w700,color:AppTheme.textPrimary)),
              Text('${s['day']} · ${s['time']}',style:GoogleFonts.dmSans(fontSize:11,color:AppTheme.textMuted)),
              const SizedBox(height:4),
              Row(children:[const Icon(Icons.person_rounded,size:11,color:AppTheme.textMuted),const SizedBox(width:4),
                Text(s['staff'] as String,style:GoogleFonts.dmSans(fontSize:11,color:AppTheme.textMuted))]),
            ])),
            GestureDetector(
              onTap:()=>setState(()=>_schedule[i]['status']=
                  _schedule[i]['status']=='Scheduled'?'Done':'Scheduled'),
              child:Container(padding:const EdgeInsets.symmetric(horizontal:10,vertical:5),
                decoration:BoxDecoration(
                    color:s['status']=='Done'?AppTheme.greenLight:AppTheme.teal.withOpacity(0.08),
                    borderRadius:BorderRadius.circular(8)),
                child:Text(s['status'] as String,style:GoogleFonts.dmSans(fontSize:11,fontWeight:FontWeight.w600,
                    color:s['status']=='Done'?AppTheme.green:AppTheme.teal)))),
          ]));
      }),
    floatingActionButton:FloatingActionButton.extended(onPressed:_showSheet,
      backgroundColor:AppTheme.teal,
      icon:const Icon(Icons.add_rounded,color:Colors.white),
      label:Text('Add Schedule',style:GoogleFonts.dmSans(color:Colors.white,fontWeight:FontWeight.w600))),
  ));

  void _showSheet(){
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => Padding(
        padding: EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
        child: Container(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
          decoration: const BoxDecoration(
            color: AppTheme.bg,
            borderRadius: BorderRadius.vertical(top: Radius.circular(28))
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  margin: const EdgeInsets.only(bottom: 14),
                  decoration: BoxDecoration(
                    color: AppTheme.border,
                    borderRadius: BorderRadius.circular(2)
                  )
                )
              ),
              Row(
                children: [
                  const Icon(Icons.cleaning_services_rounded, color: AppTheme.teal, size: 20),
                  const SizedBox(width: 8),
                  Text('Add Cleaning Schedule', style: GoogleFonts.playfairDisplay(
                    fontSize: 17,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary
                  ))
                ]
              ),
              const SizedBox(height: 14),
              _field(_areaCtrl, 'Area (e.g. Bathrooms Floor 1)', Icons.location_on_rounded),
              const SizedBox(height: 10),
              _field(_dayCtrl, 'Day (e.g. Daily / Monday)', Icons.calendar_today_rounded),
              const SizedBox(height: 10),
              _field(_timeCtrl, 'Time (e.g. 8:00 AM – 10:00 AM)', Icons.access_time_rounded),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                height: 48,
                child: ElevatedButton(
                  onPressed: _add,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.teal,
                    foregroundColor: Colors.white,
                    elevation: 0,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))
                  ),
                  child: Text('Add to Schedule', style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w600))
                )
              )
            ]
          )
        )
      )
    );
  }
}