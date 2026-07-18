import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';

class OwnerHostelProfileScreen extends StatefulWidget {
  final String ownerName;
  final String ownerEmail;
  final String hostelName;
  final String hostelType;
  final String city;
  final String address;

  const OwnerHostelProfileScreen({
    super.key,
    required this.ownerName,
    required this.ownerEmail,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    required this.address,
  });

  @override
  State<OwnerHostelProfileScreen> createState() =>
      _OwnerHostelProfileScreenState();
}

class _OwnerHostelProfileScreenState extends State<OwnerHostelProfileScreen>
    with SingleTickerProviderStateMixin {
  late final AnimationController _anim;
  late final Animation<double> _fade;
  int _selectedTab = 0;

  // Editable hostel data (pre-filled from registration)
  late String _hostelName;
  late String _hostelType;
  late String _city;
  late String _address;

  // Mock data (in real app, load from backend)
  final _facilities = ['WiFi', 'Hot Water', 'CCTV', 'Generator'];
  final _roomTypes = [
    {'type': 'Single', 'price': '15,000', 'color': AppTheme.teal},
    {'type': 'Double', 'price': '10,000', 'color': Color(0xFF185FA5)},
  ];
  bool _messIncluded = true;
  bool _cctvEnabled = true;
  bool _wardenOnsite = true;
  String _curfewTime = '10:00 PM';

  @override
  void initState() {
    super.initState();
    _hostelName = widget.hostelName;
    _hostelType = widget.hostelType;
    _city = widget.city;
    _address = widget.address;

    _anim = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 500));
    _anim.forward();
    _fade = CurvedAnimation(parent: _anim, curve: Curves.easeOut);
  }

  @override
  void dispose() {
    _anim.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        body: FadeTransition(
          opacity: _fade,
          child: CustomScrollView(
            slivers: [
              // ── Hero header ──────────────────────────────────
              SliverToBoxAdapter(child: _buildHeader()),

              // ── Tab bar ──────────────────────────────────────
              SliverPersistentHeader(
                pinned: true,
                delegate: _TabBarDelegate(_buildTabBar()),
              ),

              // ── Tab content ──────────────────────────────────
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
                sliver: SliverToBoxAdapter(
                  child: _selectedTab == 0
                      ? _buildBasicInfoTab()
                      : _selectedTab == 1
                          ? _buildFacilitiesTab()
                          : _selectedTab == 2
                              ? _buildRoomsTab()
                              : _buildSecurityTab(),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────
  Widget _buildHeader() {
    final isGirls = _hostelType.toLowerCase().contains('girl');
    final typeColor = isGirls ? const Color(0xFFE91E8C) : const Color(0xFF185FA5);

    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [AppTheme.tealDeep, AppTheme.teal],
        ),
      ),
      child: SafeArea(
        bottom: false,
        child: Column(children: [
          // Top bar
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
            child: Row(children: [
              GestureDetector(
                onTap: () => Navigator.pop(context),
                child: Container(
                  width: 38, height: 38,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.15),
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white.withOpacity(0.3))),
                  child: const Icon(Icons.arrow_back_rounded,
                      color: Colors.white, size: 18)),
              ),
              const Spacer(),
              // Live badge
              Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 12, vertical: 5),
                decoration: BoxDecoration(
                  color: AppTheme.green.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                      color: AppTheme.green.withOpacity(0.4))),
                child: Row(children: [
                  Container(
                    width: 7, height: 7,
                    decoration: const BoxDecoration(
                        color: AppTheme.tealMint,
                        shape: BoxShape.circle)),
                  const SizedBox(width: 6),
                  Text('Live on Map',
                      style: GoogleFonts.dmSans(
                          color: AppTheme.tealMint,
                          fontSize: 11,
                          fontWeight: FontWeight.w600)),
                ]),
              ),
              const SizedBox(width: 10),
              // Edit button
              GestureDetector(
                onTap: () => _showEditSheet(),
                child: Container(
                  width: 38, height: 38,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.15),
                    shape: BoxShape.circle,
                    border: Border.all(
                        color: Colors.white.withOpacity(0.3))),
                  child: const Icon(Icons.edit_rounded,
                      color: Colors.white, size: 17)),
              ),
            ]),
          ),
          const SizedBox(height: 20),

          // Hostel info
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Hostel icon
                Container(
                  width: 64, height: 64,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(18),
                    border: Border.all(
                        color: Colors.white.withOpacity(0.3))),
                  child: const Icon(Icons.apartment_rounded,
                      color: Colors.white, size: 32)),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(_hostelName,
                          style: GoogleFonts.playfairDisplay(
                              fontSize: 22,
                              fontWeight: FontWeight.w700,
                              color: Colors.white)),
                      const SizedBox(height: 5),
                      Row(children: [
                        const Icon(Icons.location_on_rounded,
                            color: Colors.white60, size: 13),
                        const SizedBox(width: 4),
                        Text('$_city · $_address',
                            style: GoogleFonts.dmSans(
                                color: Colors.white60,
                                fontSize: 11),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis),
                      ]),
                      const SizedBox(height: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: typeColor.withOpacity(0.25),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(
                              color: typeColor.withOpacity(0.4))),
                        child: Text(_hostelType,
                            style: GoogleFonts.dmSans(
                                color: Colors.white,
                                fontSize: 11,
                                fontWeight: FontWeight.w600)),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),

          // Quick stats row
          Container(
            margin: const EdgeInsets.fromLTRB(16, 0, 16, 0),
            padding: const EdgeInsets.symmetric(
                vertical: 14, horizontal: 20),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.1),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                  color: Colors.white.withOpacity(0.15))),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _statCol('Rooms', '10'),
                _vDiv(),
                _statCol('Capacity', '20'),
                _vDiv(),
                _statCol('Floors', '2'),
                _vDiv(),
                _statCol('Wardens', '1'),
              ],
            ),
          ),
          const SizedBox(height: 20),

          // Owner info chip
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Container(
              padding: const EdgeInsets.symmetric(
                  horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.08),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                    color: Colors.white.withOpacity(0.15))),
              child: Row(children: [
                Container(
                  width: 32, height: 32,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.15),
                    shape: BoxShape.circle),
                  child: const Icon(Icons.person_rounded,
                      color: Colors.white, size: 16)),
                const SizedBox(width: 10),
                Expanded(child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Text(widget.ownerName,
                      style: GoogleFonts.dmSans(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: Colors.white)),
                  Text(widget.ownerEmail,
                      style: GoogleFonts.dmSans(
                          fontSize: 11,
                          color: Colors.white60)),
                ])),
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(8)),
                  child: Text('Owner',
                      style: GoogleFonts.dmSans(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.w600)),
                ),
              ]),
            ),
          ),
        ]),
      ),
    );
  }

  Widget _statCol(String label, String value) {
    return Column(children: [
      Text(value,
          style: GoogleFonts.dmSans(
              fontSize: 18,
              fontWeight: FontWeight.w700,
              color: Colors.white)),
      Text(label,
          style: GoogleFonts.dmSans(
              fontSize: 10, color: Colors.white60)),
    ]);
  }

  Widget _vDiv() => Container(
        width: 1, height: 30,
        color: Colors.white.withOpacity(0.2));

  // ── Tab bar ───────────────────────────────────────────────────────────────
  Widget _buildTabBar() {
    final tabs = ['Basic Info', 'Facilities', 'Rooms', 'Security'];
    return Container(
      color: AppTheme.bg,
      child: Column(children: [
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          padding: const EdgeInsets.fromLTRB(16, 10, 16, 0),
          child: Row(
            children: tabs.asMap().entries.map((e) {
              final sel = _selectedTab == e.key;
              return GestureDetector(
                onTap: () => setState(() => _selectedTab = e.key),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  margin: const EdgeInsets.only(right: 8),
                  padding: const EdgeInsets.symmetric(
                      horizontal: 18, vertical: 9),
                  decoration: BoxDecoration(
                    color: sel ? AppTheme.teal : AppTheme.card,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                        color: sel ? AppTheme.teal : AppTheme.border,
                        width: 0.5)),
                  child: Text(e.value,
                      style: GoogleFonts.dmSans(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: sel
                              ? Colors.white
                              : AppTheme.textMuted)),
                ),
              );
            }).toList(),
          ),
        ),
        const SizedBox(height: 10),
        Divider(height: 0.5, color: AppTheme.border),
      ]),
    );
  }

  // ── Tab 1: Basic Info ─────────────────────────────────────────────────────
  Widget _buildBasicInfoTab() {
    return Column(children: [
      _sectionCard(
        title: 'Hostel Information',
        icon: Icons.apartment_rounded,
        onEdit: _showEditSheet,
        children: [
          _infoRow(Icons.business_rounded, 'Name', _hostelName),
          _divider(),
          _infoRow(Icons.category_rounded, 'Type', _hostelType),
          _divider(),
          _infoRow(Icons.location_city_rounded, 'City', _city),
          _divider(),
          _infoRow(Icons.pin_drop_rounded, 'Address', _address),
        ],
      ),
      const SizedBox(height: 14),
      _sectionCard(
        title: 'Owner Information',
        icon: Icons.person_rounded,
        onEdit: () {},
        children: [
          _infoRow(Icons.person_rounded, 'Owner', widget.ownerName),
          _divider(),
          _infoRow(Icons.email_rounded, 'Email', widget.ownerEmail),
        ],
      ),
      const SizedBox(height: 14),
      _sectionCard(
        title: 'Mess & Food',
        icon: Icons.restaurant_rounded,
        onEdit: () {},
        children: [
          _toggleRow('Mess included in rent', _messIncluded,
              (v) => setState(() => _messIncluded = v)),
          _divider(),
          _infoRow(Icons.set_meal_rounded, 'Meal Type', 'Both'),
          _divider(),
          _infoRow(Icons.restaurant_menu_rounded, 'Meals/day', '3'),
        ],
      ),
    ]);
  }

  // ── Tab 2: Facilities ─────────────────────────────────────────────────────
  Widget _buildFacilitiesTab() {
    final allFacilities = [
      {'name': 'WiFi',          'icon': Icons.wifi_rounded,               'enabled': true},
      {'name': 'Hot Water',     'icon': Icons.water_drop_rounded,         'enabled': true},
      {'name': 'CCTV',          'icon': Icons.videocam_rounded,           'enabled': true},
      {'name': 'Generator',     'icon': Icons.bolt_rounded,               'enabled': true},
      {'name': 'AC',            'icon': Icons.ac_unit_rounded,            'enabled': false},
      {'name': 'Study Room',    'icon': Icons.menu_book_rounded,          'enabled': false},
      {'name': 'Laundry',       'icon': Icons.local_laundry_service_rounded,'enabled': false},
      {'name': 'Security Guard','icon': Icons.security_rounded,           'enabled': false},
      {'name': 'Parking',       'icon': Icons.local_parking_rounded,      'enabled': false},
      {'name': 'Prayer Room',   'icon': Icons.mosque_rounded,             'enabled': false},
      {'name': 'Gym',           'icon': Icons.fitness_center_rounded,     'enabled': false},
      {'name': 'Common Room',   'icon': Icons.weekend_rounded,            'enabled': false},
    ];

    return Column(children: [
      // Active facilities
      _sectionCard(
        title: 'Available Facilities',
        icon: Icons.checklist_rounded,
        onEdit: () {},
        children: [
          Wrap(
            spacing: 8, runSpacing: 8,
            children: allFacilities.map((f) {
              final enabled = f['enabled'] as bool;
              final color = enabled ? AppTheme.teal : AppTheme.textMuted;
              return Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 12, vertical: 8),
                decoration: BoxDecoration(
                  color: enabled
                      ? AppTheme.teal.withOpacity(0.08)
                      : AppTheme.bgSecondary,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(
                      color: enabled
                          ? AppTheme.teal.withOpacity(0.3)
                          : AppTheme.border,
                      width: 0.5)),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  Icon(f['icon'] as IconData, size: 14, color: color),
                  const SizedBox(width: 6),
                  Text(f['name'] as String,
                      style: GoogleFonts.dmSans(
                          fontSize: 12,
                          fontWeight: FontWeight.w500,
                          color: color)),
                  if (!enabled) ...[
                    const SizedBox(width: 4),
                    Icon(Icons.close_rounded, size: 12,
                        color: AppTheme.textMuted.withOpacity(0.5)),
                  ],
                ]),
              );
            }).toList(),
          ),
        ],
      ),
    ]);
  }

  // ── Tab 3: Rooms ──────────────────────────────────────────────────────────
  Widget _buildRoomsTab() {
    return Column(children: [
      _sectionCard(
        title: 'Room Overview',
        icon: Icons.meeting_room_rounded,
        onEdit: () {},
        children: [
          _infoRow(Icons.layers_rounded, 'Total Floors', '2'),
          _divider(),
          _infoRow(Icons.meeting_room_rounded, 'Total Rooms', '10'),
          _divider(),
          _infoRow(Icons.people_rounded, 'Total Capacity', '20'),
        ],
      ),
      const SizedBox(height: 14),
      _sectionCard(
        title: 'Room Types & Pricing',
        icon: Icons.payments_rounded,
        onEdit: () {},
        children: [
          ..._roomTypes.asMap().entries.map((e) {
            final r = e.value;
            final color = r['color'] as Color;
            return Column(children: [
              if (e.key > 0) _divider(),
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8),
                child: Row(children: [
                  Container(
                    width: 36, height: 36,
                    decoration: BoxDecoration(
                      color: color.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(10)),
                    child: Icon(Icons.bed_rounded, size: 18, color: color)),
                  const SizedBox(width: 12),
                  Expanded(child: Text(
                    r['type'] as String,
                    style: GoogleFonts.dmSans(
                        fontSize: 14, fontWeight: FontWeight.w600,
                        color: AppTheme.textPrimary))),
                  Text('PKR ${r['price']}/mo',
                      style: GoogleFonts.dmSans(
                          fontSize: 13, fontWeight: FontWeight.w700,
                          color: color)),
                ]),
              ),
            ]);
          }),
        ],
      ),
    ]);
  }

  // ── Tab 4: Security ───────────────────────────────────────────────────────
  Widget _buildSecurityTab() {
    return Column(children: [
      _sectionCard(
        title: 'Safety & Security',
        icon: Icons.security_rounded,
        onEdit: () {},
        children: [
          _toggleRow('CCTV Installed', _cctvEnabled,
              (v) => setState(() => _cctvEnabled = v)),
          _divider(),
          _toggleRow('Warden Onsite', _wardenOnsite,
              (v) => setState(() => _wardenOnsite = v)),
          _divider(),
          _infoRow(Icons.lock_clock_rounded, 'Curfew Time', _curfewTime),
          _divider(),
          _infoRow(Icons.security_rounded, 'Security Guard', 'Night Shift'),
        ],
      ),
      const SizedBox(height: 14),
      _sectionCard(
        title: 'Rules & Policies',
        icon: Icons.rule_rounded,
        onEdit: () {},
        children: [
          _infoRow(Icons.how_to_reg_rounded, 'Attendance', 'Register-Based'),
          _divider(),
          _infoRow(Icons.people_rounded, 'Guest Policy', 'Not Allowed'),
          _divider(),
          _infoRow(Icons.payments_rounded, 'Payment', 'Cash · Bank Transfer'),
          _divider(),
          _infoRow(Icons.school_rounded, 'For', 'University Students'),
        ],
      ),
    ]);
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  Widget _sectionCard({
    required String title,
    required IconData icon,
    required VoidCallback onEdit,
    required List<Widget> children,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppTheme.border, width: 0.5)),
      child: Column(children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 14, 12, 10),
          child: Row(children: [
            Icon(icon, size: 15, color: AppTheme.teal),
            const SizedBox(width: 7),
            Expanded(child: Text(title,
                style: GoogleFonts.dmSans(
                    fontSize: 14, fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary))),
            GestureDetector(
              onTap: onEdit,
              child: Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(0.08),
                  borderRadius: BorderRadius.circular(8)),
                child: Row(children: [
                  const Icon(Icons.edit_rounded,
                      size: 12, color: AppTheme.teal),
                  const SizedBox(width: 4),
                  Text('Edit', style: GoogleFonts.dmSans(
                      fontSize: 11, fontWeight: FontWeight.w600,
                      color: AppTheme.teal)),
                ]),
              ),
            ),
          ]),
        ),
        Divider(height: 0.5, color: AppTheme.border, thickness: 0.5),
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 10, 16, 14),
          child: Column(children: children),
        ),
      ]),
    );
  }

  Widget _infoRow(IconData icon, String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(children: [
        Icon(icon, size: 15, color: AppTheme.textMuted),
        const SizedBox(width: 10),
        Text(label, style: GoogleFonts.dmSans(
            fontSize: 12, color: AppTheme.textMuted)),
        const Spacer(),
        Text(value, style: GoogleFonts.dmSans(
            fontSize: 13, fontWeight: FontWeight.w600,
            color: AppTheme.textPrimary),
            textAlign: TextAlign.right),
      ]),
    );
  }

  Widget _toggleRow(String label, bool value, ValueChanged<bool> onChange) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(children: [
        Expanded(child: Text(label, style: GoogleFonts.dmSans(
            fontSize: 13, color: AppTheme.textPrimary))),
        Switch(
          value: value, onChanged: onChange,
          activeColor: AppTheme.teal,
          materialTapTargetSize: MaterialTapTargetSize.shrinkWrap),
      ]),
    );
  }

  Widget _divider() =>
      Divider(height: 8, thickness: 0.5, color: AppTheme.border);

  // ── Edit sheet ────────────────────────────────────────────────────────────
  void _showEditSheet() {
    final nameCtrl = TextEditingController(text: _hostelName);
    final cityCtrl = TextEditingController(text: _city);
    final addrCtrl = TextEditingController(text: _address);

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => Padding(
        padding: EdgeInsets.only(
            bottom: MediaQuery.of(context).viewInsets.bottom),
        child: Container(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
          decoration: const BoxDecoration(
            color: AppTheme.bg,
            borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Center(child: Container(
              width: 40, height: 4,
              margin: const EdgeInsets.only(bottom: 16),
              decoration: BoxDecoration(color: AppTheme.border,
                  borderRadius: BorderRadius.circular(2)))),
            Row(children: [
              const Icon(Icons.edit_rounded,
                  color: AppTheme.teal, size: 18),
              const SizedBox(width: 8),
              Text('Edit Hostel Details',
                  style: GoogleFonts.playfairDisplay(
                      fontSize: 18, fontWeight: FontWeight.w700,
                      color: AppTheme.textPrimary)),
            ]),
            const SizedBox(height: 16),
            _editField(nameCtrl, 'Hostel Name', Icons.apartment_rounded),
            const SizedBox(height: 10),
            _editField(cityCtrl, 'City', Icons.location_city_rounded),
            const SizedBox(height: 10),
            _editField(addrCtrl, 'Address', Icons.pin_drop_rounded),
            const SizedBox(height: 18),
            SizedBox(
              width: double.infinity, height: 50,
              child: ElevatedButton(
                onPressed: () {
                  setState(() {
                    _hostelName = nameCtrl.text.trim().isNotEmpty
                        ? nameCtrl.text.trim()
                        : _hostelName;
                    _city = cityCtrl.text.trim().isNotEmpty
                        ? cityCtrl.text.trim()
                        : _city;
                    _address = addrCtrl.text.trim().isNotEmpty
                        ? addrCtrl.text.trim()
                        : _address;
                  });
                  Navigator.pop(context);
                  ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                    content: Text('Hostel details updated!',
                        style: GoogleFonts.dmSans()),
                    backgroundColor: AppTheme.teal,
                    behavior: SnackBarBehavior.floating,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12)),
                    margin: const EdgeInsets.all(16)));
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.teal,
                  foregroundColor: Colors.white, elevation: 0,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14))),
                child: Text('Save Changes',
                    style: GoogleFonts.dmSans(
                        fontSize: 15, fontWeight: FontWeight.w600)),
              ),
            ),
          ]),
        ),
      ),
    );
  }

  Widget _editField(
      TextEditingController ctrl, String hint, IconData icon) {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.card, borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.border, width: 0.5)),
      child: TextField(
        controller: ctrl,
        style: GoogleFonts.dmSans(
            fontSize: 14, color: AppTheme.textPrimary),
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: GoogleFonts.dmSans(
              color: AppTheme.textMuted, fontSize: 13),
          prefixIcon: Icon(icon, size: 18, color: AppTheme.textMuted),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(
              horizontal: 14, vertical: 13)),
      ),
    );
  }
}

// ── Tab bar sliver delegate ───────────────────────────────────────────────────
class _TabBarDelegate extends SliverPersistentHeaderDelegate {
  final Widget child;
  _TabBarDelegate(this.child);

  @override
  Widget build(BuildContext ctx, double shrinkOffset, bool overlapsContent) =>
      child;

  @override
  double get maxExtent => 52;

  @override
  double get minExtent => 52;

  @override
  bool shouldRebuild(_TabBarDelegate old) => true;
}