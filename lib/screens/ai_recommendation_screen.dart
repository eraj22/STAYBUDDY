import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../services/ai_recommendation_service.dart';
import '../model/ai_recommendation.dart';
import '../api.dart';

class AiRecommendationScreen extends StatefulWidget {
  const AiRecommendationScreen({super.key});

  @override
  State<AiRecommendationScreen> createState() =>
      _AiRecommendationScreenState();
}

class _AiRecommendationScreenState extends State<AiRecommendationScreen>
    with SingleTickerProviderStateMixin {
  // ── Form state ─────────────────────────────────────────────────
  String _gender = 'Female';
  String _department = 'Computer Science';
  String _university = 'NUST';
  String _roomType = 'Single';
  int _budgetMax = 20000;
  double _maxDist = 3.0;
  double _studyPref = 0.6;
  double _priceSens = 0.6;
  double _comfort = 0.5;
  double _noiseTol = 0.3;
  double _curfewFlex = 0.5;
  bool _needsTransport = false;
  int _topK = 5;
  final Set<String> _mustHave = {'WiFi', 'Hot Water'};

  // ── Results state ──────────────────────────────────────────────
  bool _loading = false;
  String? _error;
  List<AiRecommendation> _results = [];
  bool _hasSearched = false;
  String? _studentType;
  double? _alpha;

  late final AnimationController _anim;
  late final Animation<double> _fade;

  @override
  void initState() {
    super.initState();
    _anim = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 600));
    _fade = CurvedAnimation(parent: _anim, curve: Curves.easeOut);
    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();
    super.dispose();
  }

  // ── Fetch ───────────────────────────────────────────────────────
  Future<void> _fetch() async {
    setState(() {
      _loading = true;
      _error = null;
      _results = [];
    });

    try {
      // The live recommendation engine reads the signed-in student's profile.
      // Save the choices first so a newly registered student is immediately
      // matched with hostels and peers from the selected university.
      await Api().updateStudentProfile({
        'university': _university,
        'department': _department,
        'gender': _gender,
        'budget_max': _budgetMax,
        'max_distance_km': _maxDist,
        'study_preference': _studyPref,
      });

      final recs = await AiRecommendationService.fetchRecommendations(
        gender: _gender,
        department: _department,
        budgetMax: _budgetMax,
        maxDistanceKm: _maxDist,
        studyPreference: _studyPref,
        roomType: _roomType,
        priceSensitivity: _priceSens,
        comfortPreference: _comfort,
        noiseTolerance: _noiseTol,
        curfewFlexibility: _curfewFlex,
        needsTransport: _needsTransport,
        mustHave: _mustHave.toList(),
        topK: _topK,
      );

      setState(() {
        _results = recs;
        _hasSearched = true;
        _loading = false;
        _studentType = recs.isNotEmpty ? recs.first.studentType : null;
        _alpha = recs.isNotEmpty ? recs.first.alphaUsed : null;
      });
    } catch (e) {
      setState(() {
        _loading = false;
        _error = 'Could not load recommendations.\n'
            'Make sure the StayBuddy backend is running and you are signed in as a student.';
      });
    }
  }

  // ── UI ──────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        body: FadeTransition(
          opacity: _fade,
          child: CustomScrollView(
            slivers: [
              // Header
              SliverToBoxAdapter(child: _buildHeader()),

              // Preferences form
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
                sliver: SliverToBoxAdapter(child: _buildForm()),
              ),

              // Run button
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
                sliver: SliverToBoxAdapter(child: _buildRunButton()),
              ),

              // Results
              if (_loading)
                SliverPadding(
                  padding: const EdgeInsets.all(40),
                  sliver: SliverToBoxAdapter(child: _buildLoading()),
                )
              else if (_error != null)
                SliverPadding(
                  padding: const EdgeInsets.all(20),
                  sliver: SliverToBoxAdapter(child: _buildError()),
                )
              else if (_hasSearched && _results.isNotEmpty) ...[
                SliverPadding(
                  padding: const EdgeInsets.fromLTRB(16, 20, 16, 6),
                  sliver:
                      SliverToBoxAdapter(child: _buildIntelligenceBanner()),
                ),
                SliverPadding(
                  padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
                  sliver: SliverList(
                    delegate: SliverChildBuilderDelegate(
                      (_, i) => _buildResultCard(_results[i], i + 1),
                      childCount: _results.length,
                    ),
                  ),
                ),
              ] else if (_hasSearched)
                SliverPadding(
                  padding: const EdgeInsets.all(40),
                  sliver: SliverToBoxAdapter(child: _buildEmpty()),
                ),

              const SliverPadding(padding: EdgeInsets.only(bottom: 20)),
            ],
          ),
        ),
      ),
    );
  }

  // ── Header ──────────────────────────────────────────────────────
  Widget _buildHeader() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF0A3D3F), AppTheme.teal, Color(0xFF1D9E75)],
        ),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(children: [
                GestureDetector(
                  onTap: () => Navigator.pop(context),
                  child: Container(
                    width: 38,
                    height: 38,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.15),
                      shape: BoxShape.circle,
                      border: Border.all(
                          color: Colors.white.withOpacity(0.3), width: 1.5),
                    ),
                    child: const Icon(Icons.arrow_back_rounded,
                        color: Colors.white, size: 18),
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('AI Recommendations',
                          style: GoogleFonts.playfairDisplay(
                              fontSize: 22,
                              fontWeight: FontWeight.w700,
                              color: Colors.white)),
                      Text('Live Hybrid Engine  ·  Profile + Peer Behaviour',
                          style: GoogleFonts.dmSans(
                              color: Colors.white.withOpacity(0.6),
                              fontSize: 12)),
                    ],
                  ),
                ),
                // AI badge
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: Colors.white.withOpacity(0.3)),
                  ),
                  child: Row(children: [
                    Icon(Icons.auto_awesome_rounded,
                        color: AppTheme.tealMint, size: 14),
                    const SizedBox(width: 5),
                    Text('AI Picks',
                        style: GoogleFonts.dmSans(
                            color: Colors.white,
                            fontSize: 11,
                            fontWeight: FontWeight.w600)),
                  ]),
                ),
              ]),
              const SizedBox(height: 14),
            ],
          ),
        ),
      ),
    );
  }


  // ── Form ────────────────────────────────────────────────────────
  Widget _buildForm() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionCard(
          title: 'Your Profile',
          icon: Icons.person_outline_rounded,
          children: [
            Row(children: [
              Expanded(child: _labelDropdown(
                label: 'Gender',
                value: _gender,
                items: const ['Female', 'Male'],
                onChanged: (v) => setState(() => _gender = v!),
              )),
            ]),
            const SizedBox(height: 12),
            _labelDropdown(
              label: 'University',
              value: _university,
              items: const [
                'NUST',
                'FAST',
                'COMSATS University Islamabad',
                'Air University Islamabad',
                'Bahria University Islamabad',
                'Quaid-i-Azam University',
              ],
              onChanged: (v) => setState(() => _university = v!),
            ),
            const SizedBox(height: 12),
            _labelDropdown(
              label: 'Department',
              value: _department,
              items: const [
                'Computer Science', 'Software Engineering',
                'Electrical Engineering', 'Cyber Security',
                'Data Science', 'BBA', 'Civil Engineering', 'Social Sciences',
              ],
              onChanged: (v) => setState(() => _department = v!),
            ),
            const SizedBox(height: 12),
            Row(children: [
              Expanded(child: _labelDropdown(
                label: 'Room Type',
                value: _roomType,
                items: const ['Single', 'Double', 'Dormitory'],
                onChanged: (v) => setState(() => _roomType = v!),
              )),
              const SizedBox(width: 12),
              Expanded(child: _labelDropdown(
                label: 'Top Results',
                value: _topK.toString(),
                items: const ['3', '5', '7', '10'],
                onChanged: (v) => setState(() => _topK = int.parse(v!)),
              )),
            ]),
          ],
        ),
        const SizedBox(height: 12),

        _sectionCard(
          title: 'Budget & Distance',
          icon: Icons.payments_outlined,
          children: [
            _sliderRow(
              label: 'Max Budget',
              value: _budgetMax.toDouble(),
              min: 5000, max: 50000, divisions: 90,
              display: 'PKR ${_budgetMax.toStringAsFixed(0)}',
              onChanged: (v) => setState(() => _budgetMax = v.round()),
            ),
            const SizedBox(height: 8),
            _sliderRow(
              label: 'Max Distance',
              value: _maxDist,
              min: 0.5, max: 10.0, divisions: 19,
              display: '${_maxDist.toStringAsFixed(1)} km',
              onChanged: (v) => setState(() => _maxDist = v),
            ),
            const SizedBox(height: 8),
            Row(children: [
              const Icon(Icons.directions_bus_rounded,
                  size: 16, color: AppTheme.textMuted),
              const SizedBox(width: 8),
              Text('Needs transport nearby',
                  style: GoogleFonts.dmSans(
                      fontSize: 13, color: AppTheme.textPrimary)),
              const Spacer(),
              Switch(
                value: _needsTransport,
                onChanged: (v) => setState(() => _needsTransport = v),
                activeColor: AppTheme.teal,
              ),
            ]),
          ],
        ),
        const SizedBox(height: 12),

        _sectionCard(
          title: 'Lifestyle Preferences',
          icon: Icons.tune_rounded,
          children: [
            _sliderRow(
              label: 'Study Focus',
              value: _studyPref, min: 0, max: 1, divisions: 20,
              display: _studyPref.toStringAsFixed(2),
              color: const Color(0xFF185FA5),
              onChanged: (v) => setState(() => _studyPref = v),
            ),
            const SizedBox(height: 8),
            _sliderRow(
              label: 'Price Sensitivity',
              value: _priceSens, min: 0, max: 1, divisions: 20,
              display: _priceSens.toStringAsFixed(2),
              color: AppTheme.green,
              onChanged: (v) => setState(() => _priceSens = v),
            ),
            const SizedBox(height: 8),
            _sliderRow(
              label: 'Comfort Priority',
              value: _comfort, min: 0, max: 1, divisions: 20,
              display: _comfort.toStringAsFixed(2),
              color: const Color(0xFF7B3FC4),
              onChanged: (v) => setState(() => _comfort = v),
            ),
            const SizedBox(height: 8),
            _sliderRow(
              label: 'Noise Tolerance',
              value: _noiseTol, min: 0, max: 1, divisions: 20,
              display: _noiseTol.toStringAsFixed(2),
              color: Colors.orange.shade600,
              onChanged: (v) => setState(() => _noiseTol = v),
            ),
          ],
        ),
        const SizedBox(height: 12),

        _sectionCard(
          title: 'Must-Have Amenities',
          icon: Icons.checklist_rounded,
          children: [
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                'WiFi', 'Study Room', 'AC', 'Hot Water',
                'Laundry', 'Gym', 'Generator', 'CCTV',
                'Security Guard', 'Prayer Room', 'Cafeteria', 'Parking',
              ].map((a) {
                final sel = _mustHave.contains(a);
                return GestureDetector(
                  onTap: () => setState(() =>
                      sel ? _mustHave.remove(a) : _mustHave.add(a)),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 180),
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 7),
                    decoration: BoxDecoration(
                      color: sel ? AppTheme.teal : AppTheme.bgSecondary,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                          color: sel ? AppTheme.teal : AppTheme.border,
                          width: 0.5),
                    ),
                    child: Text(a,
                        style: GoogleFonts.dmSans(
                            fontSize: 12,
                            fontWeight: FontWeight.w500,
                            color: sel ? Colors.white : AppTheme.textPrimary)),
                  ),
                );
              }).toList(),
            ),
          ],
        ),
      ],
    );
  }

  // ── Run button ──────────────────────────────────────────────────
  Widget _buildRunButton() {
    return SizedBox(
      width: double.infinity,
      height: 54,
      child: ElevatedButton(
        onPressed: _loading ? null : _fetch,
        style: ElevatedButton.styleFrom(
          backgroundColor: AppTheme.teal,
          foregroundColor: Colors.white,
          elevation: 0,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        ),
        child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          const Icon(Icons.auto_awesome_rounded, size: 20),
          const SizedBox(width: 10),
          Text('Run AI Recommendation Engine',
              style: GoogleFonts.dmSans(
                  fontSize: 15, fontWeight: FontWeight.w600)),
        ]),
      ),
    );
  }

  // ── Loading ─────────────────────────────────────────────────────
  Widget _buildLoading() {
    return Column(
      children: [
        const SizedBox(height: 20),
        CircularProgressIndicator(color: AppTheme.teal, strokeWidth: 2),
        const SizedBox(height: 16),
        Text('Ranking live hostel recommendations...',
            style: GoogleFonts.dmSans(
                color: AppTheme.textMuted, fontSize: 14)),
        const SizedBox(height: 6),
        Text('Profile fit + peer behaviour + availability',
            style: GoogleFonts.dmSans(
                color: AppTheme.textMuted, fontSize: 12)),
      ],
    );
  }

  // ── Error ───────────────────────────────────────────────────────
  Widget _buildError() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.red.shade50,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.red.shade200, width: 0.5),
      ),
      child: Column(children: [
        Icon(Icons.error_outline_rounded,
            color: Colors.red.shade600, size: 36),
        const SizedBox(height: 12),
        Text('Recommendations unavailable',
            style: GoogleFonts.dmSans(
                fontSize: 15,
                fontWeight: FontWeight.w700,
                color: Colors.red.shade700)),
        const SizedBox(height: 8),
        Text(_error!,
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(
                color: Colors.red.shade600, fontSize: 13, height: 1.5)),
        const SizedBox(height: 16),
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Colors.red.shade100,
            borderRadius: BorderRadius.circular(10),
          ),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Start the StayBuddy backend, then try again:',
                style: GoogleFonts.dmSans(
                    fontSize: 12, fontWeight: FontWeight.w600,
                    color: Colors.red.shade800)),
            const SizedBox(height: 6),
            Text(
              'cd STAYBUDDY-main\ndocker compose up --build -d',
              style: GoogleFonts.dmSans(
                  fontSize: 12, color: Colors.red.shade700,
                  fontWeight: FontWeight.w500),
            ),
          ]),
        ),
      ]),
    );
  }

  // ── Empty ───────────────────────────────────────────────────────
  Widget _buildEmpty() {
    return Column(children: [
      Icon(Icons.search_off_rounded,
          size: 56, color: AppTheme.textMuted.withOpacity(0.4)),
      const SizedBox(height: 12),
      Text('No hostels found matching your preferences',
          style: GoogleFonts.dmSans(
              color: AppTheme.textMuted, fontSize: 14),
          textAlign: TextAlign.center),
    ]);
  }

  // ── Intelligence banner ─────────────────────────────────────────
  Widget _buildIntelligenceBanner() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.teal.withOpacity(0.06),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.teal.withOpacity(0.2), width: 0.5),
      ),
      child: Row(children: [
        Container(
          width: 38, height: 38,
          decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.12), shape: BoxShape.circle),
          child: const Icon(Icons.auto_awesome_rounded,
              color: AppTheme.teal, size: 20),
        ),
        const SizedBox(width: 12),
        Expanded(child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Personalised hostel suggestions',
              style: GoogleFonts.dmSans(
                  fontSize: 12, fontWeight: FontWeight.w700, color: AppTheme.teal),
            ),
            const SizedBox(height: 2),
            Text(
              'Based on your preferences, availability, and similar students.',
              style: GoogleFonts.dmSans(
                  fontSize: 11, color: AppTheme.textMuted),
            ),
          ],
        )),
        Text('${_results.length} results',
            style: GoogleFonts.dmSans(
                fontSize: 12,
                fontWeight: FontWeight.w700,
                color: AppTheme.teal)),
      ]),
    );
  }

  // ── Result card ─────────────────────────────────────────────────
  Widget _buildResultCard(AiRecommendation rec, int rank) {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppTheme.border, width: 0.5),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            CircleAvatar(
              radius: 18,
              backgroundColor: rank == 1 ? const Color(0xFFFFF0B3) : AppTheme.teal.withOpacity(0.12),
              child: Text(rank == 1 ? '1' : '$rank', style: GoogleFonts.dmSans(fontWeight: FontWeight.w800, color: AppTheme.teal)),
            ),
            const SizedBox(width: 12),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(rec.hostelName, style: GoogleFonts.dmSans(fontSize: 16, fontWeight: FontWeight.w800, color: AppTheme.textPrimary)),
              const SizedBox(height: 3),
              Row(children: [const Icon(Icons.location_on_outlined, size: 14, color: AppTheme.textMuted), const SizedBox(width: 3), Text(rec.area.isEmpty ? 'Location not provided' : rec.area, style: GoogleFonts.dmSans(fontSize: 11, color: AppTheme.textMuted))]),
            ])),
          ]),
          const SizedBox(height: 14),
          if (rec.reasons.isNotEmpty) ...[
            Text('Why this may suit you', style: GoogleFonts.dmSans(fontSize: 12, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
            const SizedBox(height: 6),
            ...rec.reasons.take(2).map((reason) => Padding(padding: const EdgeInsets.only(bottom: 4), child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [const Icon(Icons.check_circle_outline, size: 15, color: AppTheme.teal), const SizedBox(width: 6), Expanded(child: Text(reason, style: GoogleFonts.dmSans(fontSize: 11, color: AppTheme.textSecondary)))]))),
          ],
          const SizedBox(height: 10),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(color: AppTheme.bg, borderRadius: BorderRadius.circular(12), border: Border.all(color: AppTheme.border, width: 0.5)),
            child: Row(mainAxisAlignment: MainAxisAlignment.spaceAround, children: [
              _metric('Price', 'PKR ${rec.singleRoomPrice.toStringAsFixed(0)}/mo', Icons.payments_outlined),
              _vDivider(),
              _metric('Rating', rec.overallRating > 0 ? '${rec.overallRating.toStringAsFixed(1)} ★' : 'New', Icons.star_rounded),
            ]),
          ),
        ]),
      ),
    );
  }
  // ── Helpers ─────────────────────────────────────────────────────

  Widget _metric(String label, String value, IconData icon) {
    return Column(children: [
      Icon(icon, size: 15, color: AppTheme.textMuted),
      const SizedBox(height: 4),
      Text(value,
          style: GoogleFonts.dmSans(
              fontSize: 12, fontWeight: FontWeight.w600,
              color: AppTheme.textPrimary)),
      Text(label,
          style: GoogleFonts.dmSans(
              fontSize: 10, color: AppTheme.textMuted)),
    ]);
  }

  Widget _vDivider() => Container(
        width: 0.5, height: 36,
        color: AppTheme.border,
      );

  Widget _sectionCard({
    required String title,
    required IconData icon,
    required List<Widget> children,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.border, width: 0.5),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(icon, size: 15, color: AppTheme.teal),
          const SizedBox(width: 7),
          Text(title,
              style: GoogleFonts.dmSans(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: AppTheme.textPrimary)),
        ]),
        const SizedBox(height: 14),
        ...children,
      ]),
    );
  }

  Widget _labelDropdown({
    required String label,
    required String value,
    required List<String> items,
    required ValueChanged<String?> onChanged,
  }) {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(label,
          style: GoogleFonts.dmSans(
              fontSize: 11,
              color: AppTheme.textMuted,
              fontWeight: FontWeight.w500)),
      const SizedBox(height: 4),
      Container(
        padding: const EdgeInsets.symmetric(horizontal: 12),
        decoration: BoxDecoration(
          color: AppTheme.bg,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: AppTheme.border, width: 0.5),
        ),
        child: DropdownButtonHideUnderline(
          child: DropdownButton<String>(
            value: value,
            isExpanded: true,
            style: GoogleFonts.dmSans(
                fontSize: 13, color: AppTheme.textPrimary),
            items: items
                .map((i) => DropdownMenuItem(value: i, child: Text(i)))
                .toList(),
            onChanged: onChanged,
          ),
        ),
      ),
    ]);
  }

  Widget _sliderRow({
    required String label,
    required double value,
    required double min,
    required double max,
    required int divisions,
    required String display,
    required ValueChanged<double> onChanged,
    Color? color,
  }) {
    return Column(children: [
      Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
        Text(label,
            style: GoogleFonts.dmSans(
                fontSize: 12, color: AppTheme.textSecondary)),
        Text(display,
            style: GoogleFonts.dmSans(
                fontSize: 12,
                fontWeight: FontWeight.w700,
                color: color ?? AppTheme.teal)),
      ]),
      SliderTheme(
        data: SliderTheme.of(context).copyWith(
          activeTrackColor: color ?? AppTheme.teal,
          thumbColor: color ?? AppTheme.teal,
          inactiveTrackColor: AppTheme.bgSecondary,
          overlayColor: (color ?? AppTheme.teal).withOpacity(0.1),
          trackHeight: 3,
          thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 6),
        ),
        child: Slider(
          value: value,
          min: min,
          max: max,
          divisions: divisions,
          onChanged: onChanged,
        ),
      ),
    ]);
  }
}
