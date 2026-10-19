import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import 'booking_screen.dart';

const String _base = 'http://127.0.0.1:5000';

class AiRoomSearchScreen extends StatefulWidget {
  final int hostelId;
  final String hostelName;
  final String city;
  final String address;
  final String studentName;
  final String studentPhone;
  final String studentEmail;
  final String studentCnic;
  final String studentUniversity;

  const AiRoomSearchScreen({
    super.key,
    this.hostelId = 0,
    this.hostelName = 'StayBuddy',
    this.city = '',
    this.address = '',
    this.studentName = '',
    this.studentPhone = '',
    this.studentEmail = '',
    this.studentCnic = '',
    this.studentUniversity = '',
  });

  @override
  State<AiRoomSearchScreen> createState() => _AiRoomSearchScreenState();
}

class _AiRoomSearchScreenState extends State<AiRoomSearchScreen>
    with SingleTickerProviderStateMixin {

  String? _selectedId;
  Map<String, dynamic>? _profile;
  List<Map<String, dynamic>> _allResidents = [];
  bool _loadingResidents = false;
  bool _loadingProfile   = false;
  bool _loadingResult    = false;
  String? _error;

  int _tab = 0;
  List<dynamic> _results = [];
  bool _hasResult = false;

  String _filterGender   = 'Any';
  String _filterRoomType = 'Any';

  late final TabController _tabCtrl;

  final _tabs = const [
    {'label': 'Find Rooms',       'icon': Icons.bed_rounded},
    {'label': 'Roommates',        'icon': Icons.people_rounded},
    {'label': 'Filter Rooms',     'icon': Icons.tune_rounded},
    {'label': 'Cultural Match',   'icon': Icons.public_rounded},
    {'label': 'Dept Match',       'icon': Icons.school_rounded},
    {'label': 'Hometown Match',   'icon': Icons.location_city_rounded},
  ];

  @override
  void initState() {
    super.initState();
    _tabCtrl = TabController(length: _tabs.length, vsync: this);
    _tabCtrl.addListener(() {
      if (!_tabCtrl.indexIsChanging) {
        setState(() { _tab = _tabCtrl.index; _results = []; _hasResult = false; _error = null; });
      }
    });
    _fetchAllResidents();
  }

  @override
  void dispose() { _tabCtrl.dispose(); super.dispose(); }

  Future<void> _fetchAllResidents() async {
    setState(() => _loadingResidents = true);
    try {
      final r = await http.get(Uri.parse('$_base/api/all_students'));
      final list = jsonDecode(r.body) as List;
      setState(() { _allResidents = list.cast<Map<String, dynamic>>(); _loadingResidents = false; });
    } catch (_) { setState(() => _loadingResidents = false); }
  }

  Future<void> _loadProfile() async {
    if (_selectedId == null) return;
    setState(() { _loadingProfile = true; _profile = null; _error = null; _results = []; _hasResult = false; });
    try {
      final r = await http.get(Uri.parse('$_base/api/student/$_selectedId'));
      if (r.statusCode == 200) {
        setState(() { _profile = jsonDecode(r.body); _loadingProfile = false; });
      } else {
        setState(() { _error = 'Resident not found'; _loadingProfile = false; });
      }
    } catch (_) {
      setState(() { _error = 'Cannot connect to AI server'; _loadingProfile = false; });
    }
  }

  Future<Map<String, dynamic>> _post(String path, Map body) async {
    final r = await http.post(Uri.parse('$_base$path'),
        headers: {'Content-Type': 'application/json'}, body: jsonEncode(body));
    if (r.statusCode != 200) throw Exception('Server error: ${r.body}');
    final d = jsonDecode(r.body) as Map<String, dynamic>;
    if (d['error'] != null) throw Exception(d['error']);
    return d;
  }

  Future<void> _runTab() async {
    if (_selectedId == null || _profile == null) return;
    setState(() { _loadingResult = true; _error = null; _results = []; _hasResult = false; });
    try {
      Map<String, dynamic> data;
      switch (_tab) {
        case 0: data = await _post('/api/find_rooms',      {'resident_id': _selectedId}); break;
        case 1: data = await _post('/api/find_roommates',  {'resident_id': _selectedId, 'top_k': 10}); break;
        case 2:
          final prefs = <String, dynamic>{};
          if (_filterGender   != 'Any') prefs['gender']    = _filterGender;
          if (_filterRoomType != 'Any') prefs['room_type'] = _filterRoomType;
          data = await _post('/api/filter_rooms', {'resident_id': _selectedId, 'preferences': prefs}); break;
        case 3: data = await _post('/api/find_by_culture',    {'resident_id': _selectedId}); break;
        case 4: data = await _post('/api/find_by_department', {'resident_id': _selectedId}); break;
        case 5: data = await _post('/api/find_by_hometown',   {'resident_id': _selectedId}); break;
        default: data = {};
      }
      final list = data['rooms'] ?? data['roommates'] ?? data['matches'] ?? [];
      setState(() { _results = list; _hasResult = true; _loadingResult = false; });
    } catch (e) {
      setState(() { _error = e.toString().replaceAll('Exception: ', ''); _loadingResult = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        body: Column(children: [
          _buildHeader(),
          _buildResidentPicker(),
          if (_profile != null) _buildProfileCard(),
          if (_profile != null) _buildTabBar(),
          if (_profile != null && _tab == 2) _buildFilterOptions(),
          Expanded(child: _buildBody()),
        ]),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight,
            colors: [Color(0xFF063E40), AppTheme.teal, Color(0xFF1D9E75)])),
      child: SafeArea(bottom: false,
        child: Padding(padding: const EdgeInsets.fromLTRB(16, 14, 16, 18),
          child: Row(children: [
            GestureDetector(onTap: () => Navigator.pop(context),
              child: Container(width: 38, height: 38,
                decoration: BoxDecoration(color: Colors.white.withOpacity(0.15),
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white.withOpacity(0.3))),
                child: const Icon(Icons.arrow_back_rounded, color: Colors.white, size: 18))),
            const SizedBox(width: 14),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('AI Room Matching', style: GoogleFonts.playfairDisplay(
                  fontSize: 20, fontWeight: FontWeight.w700, color: Colors.white)),
              Text('Select ID → Load Profile → Run Match',
                  style: GoogleFonts.dmSans(fontSize: 11, color: Colors.white.withOpacity(0.6))),
            ])),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              decoration: BoxDecoration(color: Colors.white.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: Colors.white.withOpacity(0.3))),
              child: Row(children: [
                Icon(Icons.psychology_rounded, color: AppTheme.tealMint, size: 14),
                const SizedBox(width: 5),
                Text('ML Model', style: GoogleFonts.dmSans(
                    color: Colors.white, fontSize: 11, fontWeight: FontWeight.w600)),
              ])),
          ]))));
  }

  Widget _buildResidentPicker() {
    return Container(
      color: AppTheme.card,
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
      child: Row(children: [
        Expanded(
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            decoration: BoxDecoration(color: AppTheme.bg,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AppTheme.border)),
            child: _loadingResidents
                ? const Padding(padding: EdgeInsets.symmetric(vertical: 14),
                    child: Center(child: SizedBox(width: 18, height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2, color: AppTheme.teal))))
                : DropdownButtonHideUnderline(
                    child: DropdownButton<String>(
                      value: _selectedId, isExpanded: true,
                      hint: Text('-- Select Your Resident ID --',
                          style: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 13)),
                      style: GoogleFonts.dmSans(fontSize: 13, color: AppTheme.textPrimary),
                      items: _allResidents.map((r) {
                        final id   = r['id'].toString();
                        final dept = r['department']?.toString() ?? '';
                        final gen  = r['gender']?.toString() ?? '';
                        final isPro = r['is_professional'] == true;
                        return DropdownMenuItem(value: id,
                          child: Text('$id  ·  $gen  ·  ${isPro ? (r['job_title'] ?? 'Pro') : dept}',
                              style: GoogleFonts.dmSans(fontSize: 12)));
                      }).toList(),
                      onChanged: (v) => setState(() {
                        _selectedId = v; _profile = null; _results = []; _hasResult = false;
                      }),
                    ))),
        ),
        const SizedBox(width: 10),
        SizedBox(height: 48,
          child: ElevatedButton(
            onPressed: _selectedId == null ? null : _loadProfile,
            style: ElevatedButton.styleFrom(backgroundColor: AppTheme.teal,
                foregroundColor: Colors.white, elevation: 0,
                padding: const EdgeInsets.symmetric(horizontal: 16),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))),
            child: _loadingProfile
                ? const SizedBox(width: 18, height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                : Text('Load Profile',
                    style: GoogleFonts.dmSans(fontSize: 13, fontWeight: FontWeight.w600)),
          )),
      ]),
    );
  }

  Widget _buildProfileCard() {
    final p = _profile!;
    final isPro = p['is_professional'] == true;
    final color = isPro ? const Color(0xFF7B3FC4) : AppTheme.teal;
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 10, 16, 0),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(color: color.withOpacity(0.06),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: color.withOpacity(0.25))),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Container(width: 42, height: 42,
              decoration: BoxDecoration(color: color.withOpacity(0.12), shape: BoxShape.circle),
              child: Icon(isPro ? Icons.work_rounded : Icons.school_rounded, color: color, size: 20)),
          const SizedBox(width: 12),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(_selectedId ?? '',
                style: GoogleFonts.dmSans(fontSize: 15, fontWeight: FontWeight.w700, color: color)),
            Text('${p['gender'] ?? ''}  ·  ${p['department'] ?? ''}  ·  ${p['university'] ?? ''}',
                style: GoogleFonts.dmSans(fontSize: 11, color: AppTheme.textMuted)),
          ])),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(color: color.withOpacity(0.12), borderRadius: BorderRadius.circular(8)),
            child: Text(isPro ? 'Professional' : 'Student',
                style: GoogleFonts.dmSans(fontSize: 10, fontWeight: FontWeight.w700, color: color))),
        ]),
        const SizedBox(height: 10),
        Wrap(spacing: 8, runSpacing: 6, children: [
          _chip(Icons.home_rounded,             '${p['home_city'] ?? 'N/A'}'),
          _chip(Icons.public_rounded,           '${p['ethnicity'] ?? 'N/A'}'),
          _chip(Icons.bedtime_rounded,          '${p['sleep_schedule'] ?? 'N/A'}'),
          _chip(Icons.cleaning_services_rounded,'Clean: ${p['cleanliness_level'] ?? 'N/A'}/5'),
          _chip(Icons.restaurant_rounded,       '${p['food_preference'] ?? 'N/A'}'),
          _chip(Icons.attach_money_rounded,     'PKR ${p['budget_min'] ?? 0}–${p['budget_max'] ?? 0}'),
        ]),
      ]),
    );
  }

  Widget _chip(IconData icon, String label) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
    decoration: BoxDecoration(color: AppTheme.card,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: AppTheme.border, width: 0.5)),
    child: Row(mainAxisSize: MainAxisSize.min, children: [
      Icon(icon, size: 11, color: AppTheme.teal),
      const SizedBox(width: 5),
      Flexible(child: Text(label, style: GoogleFonts.dmSans(fontSize: 11, color: AppTheme.textSecondary),
          overflow: TextOverflow.ellipsis)),
    ]));

  Widget _buildTabBar() {
    return Container(
      margin: const EdgeInsets.only(top: 12),
      height: 40,
      child: TabBar(
        controller: _tabCtrl, isScrollable: true,
        labelColor: AppTheme.teal, unselectedLabelColor: AppTheme.textMuted,
        indicatorColor: AppTheme.teal, indicatorWeight: 2.5,
        labelStyle: GoogleFonts.dmSans(fontSize: 12, fontWeight: FontWeight.w600),
        unselectedLabelStyle: GoogleFonts.dmSans(fontSize: 12),
        tabAlignment: TabAlignment.start,
        padding: const EdgeInsets.symmetric(horizontal: 12),
        tabs: _tabs.map((t) => Tab(
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            Icon(t['icon'] as IconData, size: 14),
            const SizedBox(width: 5),
            Text(t['label'] as String),
          ]))).toList(),
      ),
    );
  }

  Widget _buildFilterOptions() {
    return Container(
      color: AppTheme.card,
      padding: const EdgeInsets.fromLTRB(16, 10, 16, 10),
      child: Row(children: [
        Expanded(child: _miniDrop('Gender', _filterGender, ['Any','Female','Male'],
            (v) => setState(() => _filterGender = v!))),
        const SizedBox(width: 10),
        Expanded(child: _miniDrop('Room Type', _filterRoomType,
            ['Any','Single','Double','Triple','Dormitory'],
            (v) => setState(() => _filterRoomType = v!))),
      ]),
    );
  }

  Widget _miniDrop(String label, String value, List<String> items, ValueChanged<String?> cb) {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(label, style: GoogleFonts.dmSans(fontSize: 10, color: AppTheme.textMuted, fontWeight: FontWeight.w600)),
      const SizedBox(height: 3),
      Container(
        padding: const EdgeInsets.symmetric(horizontal: 10),
        decoration: BoxDecoration(color: AppTheme.bg,
            borderRadius: BorderRadius.circular(8), border: Border.all(color: AppTheme.border, width: 0.5)),
        child: DropdownButtonHideUnderline(
          child: DropdownButton<String>(value: value, isExpanded: true,
            style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textPrimary),
            items: items.map((i) => DropdownMenuItem(value: i, child: Text(i))).toList(),
            onChanged: cb))),
    ]);
  }

  Widget _buildBody() {
    if (_profile == null) {
      return Center(child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        Icon(Icons.person_search_rounded, size: 56, color: AppTheme.textMuted.withOpacity(0.3)),
        const SizedBox(height: 14),
        Text('Select your Resident ID above and tap Load Profile',
            style: GoogleFonts.dmSans(fontSize: 13, color: AppTheme.textMuted), textAlign: TextAlign.center),
        const SizedBox(height: 6),
        Text('STU-001 to STU-200  ·  JOB-001 to JOB-150',
            style: GoogleFonts.dmSans(fontSize: 11, color: AppTheme.textMuted.withOpacity(0.5))),
      ]));
    }
    return Column(children: [
      Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
        child: SizedBox(width: double.infinity, height: 48,
          child: ElevatedButton(
            onPressed: _loadingResult ? null : _runTab,
            style: ElevatedButton.styleFrom(backgroundColor: AppTheme.teal,
                foregroundColor: Colors.white, elevation: 0,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14))),
            child: _loadingResult
                ? Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                    const SizedBox(width: 18, height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)),
                    const SizedBox(width: 10),
                    Text('Running AI Matcher...', style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w600)),
                  ])
                : Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                    const Icon(Icons.auto_awesome_rounded, size: 18),
                    const SizedBox(width: 8),
                    Text('Run: ${(_tabs[_tab]['label'] as String)}',
                        style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w600)),
                  ]),
          )),
      ),
      Expanded(child: _buildResults()),
    ]);
  }

  Widget _buildResults() {
    if (_error != null) return _buildError();
    if (!_hasResult) return const SizedBox();
    if (_results.isEmpty) return Center(child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
      Icon(Icons.search_off_rounded, size: 48, color: AppTheme.textMuted.withOpacity(0.3)),
      const SizedBox(height: 12),
      Text('No matches found', style: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 14)),
    ]));

    return ListView.builder(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 32),
      itemCount: _results.length,
      itemBuilder: (_, i) {
        final item = _results[i] as Map<String, dynamic>;
        final isRoom = item.containsKey('room_id') || item.containsKey('room_type');
        return isRoom ? _buildRoomCard(item, i + 1) : _buildPersonCard(item, i + 1);
      },
    );
  }

  Widget _buildRoomCard(Map<String, dynamic> room, int rank) {
    final roomId   = room['room_id']?.toString() ?? 'Room';
    final roomType = room['room_type']?.toString() ?? room['type']?.toString() ?? '-';
    final score    = (room['overall_score'] ?? room['similarity_score'] ?? 0.0) as num;
    final pct      = (score * 100).round();
    final capacity = (room['capacity'] ?? 0) as num;
    final occupants= (room['current_occupants'] as List?)?.length ?? 0;
    final vacancies= capacity.toInt() - occupants;
    final reasons  = (room['recommendation_reasons'] as List?) ?? (room['reasons'] as List?) ?? [];
    final scoreColor = pct >= 70 ? AppTheme.green : pct >= 45 ? AppTheme.amber : Colors.red.shade500;

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      decoration: BoxDecoration(color: AppTheme.card, borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppTheme.border, width: 0.5),
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2))]),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Padding(padding: const EdgeInsets.fromLTRB(14, 14, 14, 10),
          child: Row(children: [
            Container(width: 36, height: 36,
              decoration: BoxDecoration(
                color: rank == 1 ? const Color(0xFFFFF0B3) : AppTheme.teal.withOpacity(0.1),
                shape: BoxShape.circle,
                border: Border.all(color: rank == 1 ? AppTheme.amber : AppTheme.teal.withOpacity(0.2), width: 0.5)),
              child: Center(child: Text(rank == 1 ? '🥇' : '#$rank',
                  style: GoogleFonts.dmSans(fontSize: rank == 1 ? 16 : 12, fontWeight: FontWeight.w700, color: AppTheme.teal)))),
            const SizedBox(width: 12),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Room $roomId', style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
              Text('$roomType  ·  $vacancies vacancies', style: GoogleFonts.dmSans(fontSize: 11, color: AppTheme.textMuted)),
            ])),
            Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
              Text('$pct%', style: GoogleFonts.dmSans(fontSize: 22, fontWeight: FontWeight.w700, color: scoreColor)),
              Text('match', style: GoogleFonts.dmSans(fontSize: 10, color: AppTheme.textMuted)),
            ]),
          ])),
        Padding(padding: const EdgeInsets.symmetric(horizontal: 14),
          child: ClipRRect(borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(value: score.toDouble().clamp(0.0, 1.0),
                backgroundColor: AppTheme.bgSecondary,
                valueColor: AlwaysStoppedAnimation(scoreColor), minHeight: 5))),
        const SizedBox(height: 10),
        Container(
          margin: const EdgeInsets.fromLTRB(14, 0, 14, 0),
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(color: AppTheme.bg, borderRadius: BorderRadius.circular(10),
              border: Border.all(color: AppTheme.border, width: 0.5)),
          child: Row(mainAxisAlignment: MainAxisAlignment.spaceAround, children: [
            _metric('Type', roomType, Icons.meeting_room_rounded), _vDiv(),
            _metric('Capacity', '${capacity.toInt()} beds', Icons.people_rounded), _vDiv(),
            _metric('Free', '$vacancies left', Icons.door_front_door_rounded),
          ])),
        if (reasons.isNotEmpty)
          Padding(padding: const EdgeInsets.fromLTRB(14, 10, 14, 0),
            child: Wrap(spacing: 6, runSpacing: 6,
              children: reasons.take(3).map((r) => Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(color: AppTheme.teal.withOpacity(0.06),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: AppTheme.teal.withOpacity(0.18), width: 0.5)),
                child: Text(r.toString(), style: GoogleFonts.dmSans(fontSize: 10, color: AppTheme.teal)),
              )).toList())),
        Padding(padding: const EdgeInsets.fromLTRB(14, 12, 14, 14),
          child: SizedBox(width: double.infinity, height: 44,
            child: ElevatedButton(
              onPressed: vacancies > 0 ? () => Navigator.push(context, MaterialPageRoute(
                  builder: (_) => BookingScreen(hostelId: widget.hostelId,
                      hostelName: widget.hostelName, city: widget.city, address: widget.address,
                      studentName: widget.studentName, studentPhone: widget.studentPhone,
                      studentEmail: widget.studentEmail, studentCnic: widget.studentCnic,
                      studentUniversity: widget.studentUniversity))) : null,
              style: ElevatedButton.styleFrom(
                  backgroundColor: vacancies > 0 ? AppTheme.teal : AppTheme.bgSecondary,
                  foregroundColor: Colors.white, elevation: 0,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))),
              child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                Icon(vacancies > 0 ? Icons.hotel_rounded : Icons.block_rounded, size: 16),
                const SizedBox(width: 8),
                Text(vacancies > 0 ? 'Book This Room' : 'No Vacancies',
                    style: GoogleFonts.dmSans(fontSize: 13, fontWeight: FontWeight.w600)),
              ])))),
      ]));
  }

  Widget _buildPersonCard(Map<String, dynamic> p, int rank) {
    final id    = p['student_id']?.toString() ?? 'Unknown';
    final score = (p['overall_score'] ?? p['similarity_score'] ?? 0.0) as num;
    final pct   = (score * 100).round();
    final dept  = p['department']?.toString() ?? 'N/A';
    final city  = p['home_city']?.toString() ?? 'N/A';
    final eth   = p['ethnicity']?.toString() ?? 'N/A';
    final sleep = p['sleep_schedule']?.toString() ?? 'N/A';
    final pers  = p['personality']?.toString() ?? '';
    final uni   = p['university']?.toString() ?? '';
    final isPro = p['is_professional'] == true;
    final job   = p['job_title']?.toString() ?? '';
    final scoreColor = pct >= 70 ? AppTheme.green : pct >= 45 ? AppTheme.amber : Colors.red.shade500;

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(color: AppTheme.card, borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppTheme.border, width: 0.5),
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 6, offset: const Offset(0, 2))]),
      child: Padding(padding: const EdgeInsets.all(14),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Container(width: 34, height: 34,
              decoration: BoxDecoration(
                color: rank == 1 ? const Color(0xFFFFF0B3) : AppTheme.teal.withOpacity(0.08),
                shape: BoxShape.circle,
                border: Border.all(color: rank == 1 ? AppTheme.amber : AppTheme.teal.withOpacity(0.2), width: 0.5)),
              child: Center(child: Text(rank == 1 ? '🥇' : '#$rank',
                  style: GoogleFonts.dmSans(fontSize: rank == 1 ? 14 : 11, fontWeight: FontWeight.w700, color: AppTheme.teal)))),
            const SizedBox(width: 10),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Row(children: [
                Text(id, style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                  decoration: BoxDecoration(
                    color: isPro ? const Color(0xFF7B3FC4).withOpacity(0.1) : AppTheme.teal.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(6)),
                  child: Text(isPro ? 'Pro' : 'Student',
                      style: GoogleFonts.dmSans(fontSize: 9, fontWeight: FontWeight.w700,
                          color: isPro ? const Color(0xFF7B3FC4) : AppTheme.teal))),
              ]),
              Text(isPro ? '$job · $dept' : '$dept${uni.isNotEmpty ? ' · $uni' : ''}',
                  style: GoogleFonts.dmSans(fontSize: 11, color: AppTheme.textMuted)),
            ])),
            Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
              Text('$pct%', style: GoogleFonts.dmSans(fontSize: 20, fontWeight: FontWeight.w700, color: scoreColor)),
              Text('compat.', style: GoogleFonts.dmSans(fontSize: 9, color: AppTheme.textMuted)),
            ]),
          ]),
          const SizedBox(height: 8),
          ClipRRect(borderRadius: BorderRadius.circular(3),
            child: LinearProgressIndicator(value: score.toDouble().clamp(0.0, 1.0),
                backgroundColor: AppTheme.bgSecondary,
                valueColor: AlwaysStoppedAnimation(scoreColor), minHeight: 4)),
          const SizedBox(height: 10),
          Wrap(spacing: 6, runSpacing: 6, children: [
            if (city != 'N/A') _tag(Icons.location_city_rounded, city),
            if (eth  != 'N/A') _tag(Icons.public_rounded, eth),
            if (sleep != 'N/A') _tag(Icons.bedtime_rounded, sleep),
            if (pers.isNotEmpty) _tag(Icons.psychology_rounded, pers),
          ]),
        ])));
  }

  Widget _buildError() => Padding(padding: const EdgeInsets.all(20),
    child: Container(padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Colors.red.shade50,
          borderRadius: BorderRadius.circular(14), border: Border.all(color: Colors.red.shade200)),
      child: Column(children: [
        Icon(Icons.error_outline_rounded, color: Colors.red.shade600, size: 36),
        const SizedBox(height: 10),
        Text('Error', style: GoogleFonts.dmSans(fontSize: 14, fontWeight: FontWeight.w700, color: Colors.red.shade700)),
        const SizedBox(height: 8),
        Text(_error!, textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(color: Colors.red.shade600, fontSize: 12, height: 1.5)),
        const SizedBox(height: 12),
        Container(padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(color: Colors.red.shade100, borderRadius: BorderRadius.circular(8)),
          child: Text('cd room_matcher\npython web_app.py',
              style: GoogleFonts.dmSans(fontSize: 12, fontWeight: FontWeight.w600, color: Colors.red.shade800))),
      ])));

  Widget _metric(String label, String value, IconData icon) => Column(children: [
    Icon(icon, size: 14, color: AppTheme.textMuted), const SizedBox(height: 3),
    Text(value, style: GoogleFonts.dmSans(fontSize: 12, fontWeight: FontWeight.w600, color: AppTheme.textPrimary)),
    Text(label, style: GoogleFonts.dmSans(fontSize: 9, color: AppTheme.textMuted)),
  ]);

  Widget _vDiv() => Container(width: 0.5, height: 32, color: AppTheme.border);

  Widget _tag(IconData icon, String text) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
    decoration: BoxDecoration(color: AppTheme.bgSecondary, borderRadius: BorderRadius.circular(6),
        border: Border.all(color: AppTheme.border, width: 0.5)),
    child: Row(mainAxisSize: MainAxisSize.min, children: [
      Icon(icon, size: 11, color: AppTheme.teal), const SizedBox(width: 4),
      Text(text, style: GoogleFonts.dmSans(fontSize: 11, color: AppTheme.textSecondary)),
    ]));
}