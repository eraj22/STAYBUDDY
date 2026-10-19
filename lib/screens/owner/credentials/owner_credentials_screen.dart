import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../theme.dart';
import '../../../api.dart';
import '../../../widgets/owner_feature_frame.dart';
import '../../../widgets/responsive_shell.dart';

class OwnerCredentialsScreen extends StatefulWidget {
  final String hostelName;
  final VoidCallback? onBackToDashboard;

  const OwnerCredentialsScreen({
    super.key,
    required this.hostelName,
    this.onBackToDashboard,
  });

  @override
  State<OwnerCredentialsScreen> createState() => _OwnerCredentialsScreenState();
}

class _OwnerCredentialsScreenState extends State<OwnerCredentialsScreen> {
  final TextEditingController _searchCtrl = TextEditingController();

  String _selectedType = 'All';

  final List<String> _types = ['All', 'Students', 'Parents', 'Wardens'];

  final List<Map<String, dynamic>> _credentials = [];
  List<Map<String, dynamic>> _approvedStudents = [];
  bool _loadingCredentials = true;

  @override
  void initState() {
    super.initState();
    _loadCredentials();
  }

  Future<void> _loadCredentials() async {
    try {
      final results = await Future.wait([
        Api().getOwnerStudents(),
        Api().getOwnerParentCredentials(),
      ]);
      final students = results[0] as List<dynamic>;
      final parents = results[1] as List<dynamic>;
      if (!mounted) return;
      setState(() {
        _credentials
          ..clear()
          ..addAll(
            students.whereType<Map>().map((raw) {
              final item = Map<String, dynamic>.from(raw);
              final email = item['email']?.toString() ?? '';
              final hasRoom =
                  item['room_number']?.toString().trim().isNotEmpty == true;
              final status =
                  item['booking_status']?.toString() == 'confirmed' && hasRoom
                  ? 'Active'
                  : 'Pending';
              return <String, dynamic>{
                'userId': item['id'],
                'name': item['name'] ?? 'Student',
                'role': 'Student',
                'email': email,
                'username': email.split('@').first,
                'password': 'Hidden (shown once at generation)',
                'status': status,
                'lastLogin': 'Not available',
                'temporary': true,
              };
            }),
          )
          ..addAll(
            parents.whereType<Map>().map((raw) {
              final item = Map<String, dynamic>.from(raw);
              final email = item['email']?.toString() ?? '';
              return <String, dynamic>{
                'userId': item['id'],
                'name': item['name'] ?? 'Parent',
                'role': 'Parent',
                'email': email,
                'username': email.split('@').first,
                'password': 'Hidden',
                'status': 'Active',
                'lastLogin': 'Not available',
                'temporary': false,
              };
            }),
          );
        _approvedStudents = students
            .whereType<Map>()
            .map((raw) => Map<String, dynamic>.from(raw))
            .where(
              (item) =>
                  (item['booking_status']?.toString() == 'confirmed' ||
                      item['hostel_credential_id'] != null) &&
                  item['booking_id'] != null,
            )
            .toList();
        _loadingCredentials = false;
      });
    } catch (_) {
      if (mounted) setState(() => _loadingCredentials = false);
    }
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  List<Map<String, dynamic>> get _filteredCredentials {
    final query = _searchCtrl.text.trim().toLowerCase();

    return _credentials.where((item) {
      final name = item['name'].toString().toLowerCase();
      final email = item['email'].toString().toLowerCase();
      final username = item['username'].toString().toLowerCase();
      final role = item['role'].toString();

      final matchesSearch =
          query.isEmpty ||
          name.contains(query) ||
          email.contains(query) ||
          username.contains(query);

      final matchesType =
          _selectedType == 'All' ||
          (_selectedType == 'Students' && role == 'Student') ||
          (_selectedType == 'Parents' && role == 'Parent') ||
          (_selectedType == 'Wardens' && role == 'Warden');

      return matchesSearch && matchesType;
    }).toList();
  }

  int get _total => _credentials.length;

  int get _active => _credentials.where((e) => e['status'] == 'Active').length;

  int get _pending =>
      _credentials.where((e) => e['status'] == 'Pending').length;

  int get _disabled =>
      _credentials.where((e) => e['status'] == 'Disabled').length;

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        floatingActionButton: _buildFab(),
        body: OwnerFeatureFrame(
          child: CustomScrollView(
            physics: const BouncingScrollPhysics(),
            slivers: [
              SliverToBoxAdapter(child: _buildHeader()),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 18, 18, 0),
                sliver: SliverToBoxAdapter(child: _buildSummary()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 22, 18, 0),
                sliver: SliverToBoxAdapter(child: _buildSearch()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 14, 18, 0),
                sliver: SliverToBoxAdapter(child: _buildTypeFilters()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                sliver: SliverToBoxAdapter(child: _buildSectionHeader()),
              ),

              _buildCredentialsSliver(),

              const SliverToBoxAdapter(child: SizedBox(height: 110)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF073F42), Color(0xFF0A696D), Color(0xFF0A797B)],
        ),
        borderRadius: BorderRadius.vertical(bottom: Radius.circular(32)),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(18, 14, 18, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  _circleButton(
                    icon: Icons.arrow_back_rounded,
                    onTap: _goBackToDashboard,
                  ),
                  const Spacer(),
                  _circleButton(
                    icon: Icons.more_horiz_rounded,
                    onTap: _showMoreMenu,
                  ),
                ],
              ),

              const SizedBox(height: 22),

              Text(
                'Login Credentials',
                style: GoogleFonts.playfairDisplay(
                  fontSize: 29,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),

              const SizedBox(height: 5),

              Text(
                widget.hostelName,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.dmSans(
                  fontSize: 12.5,
                  color: Colors.white.withOpacity(0.65),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _circleButton({required IconData icon, required VoidCallback onTap}) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        customBorder: const CircleBorder(),
        child: Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.13),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white.withOpacity(0.18)),
          ),
          child: Icon(icon, color: Colors.white, size: 20),
        ),
      ),
    );
  }

  void _goBackToDashboard() {
    if (widget.onBackToDashboard != null) {
      widget.onBackToDashboard!();
      return;
    }
    Navigator.of(context).maybePop();
  }

  Widget _buildSummary() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: OwnerFeaturePalette.border, width: 0.6),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.035),
            blurRadius: 16,
            offset: const Offset(0, 7),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Credential Overview',
            style: GoogleFonts.dmSans(
              fontSize: 14,
              fontWeight: FontWeight.w700,
              color: OwnerFeaturePalette.text,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: _summaryItem(
                  value: '$_total',
                  label: 'Total',
                  icon: Icons.password_rounded,
                  color: AppTheme.teal,
                ),
              ),
              _divider(),
              Expanded(
                child: _summaryItem(
                  value: '$_active',
                  label: 'Active',
                  icon: Icons.verified_user_rounded,
                  color: const Color(0xFF3D7B53),
                ),
              ),
              _divider(),
              Expanded(
                child: _summaryItem(
                  value: '$_pending',
                  label: 'Pending',
                  icon: Icons.schedule_rounded,
                  color: const Color(0xFFC07421),
                ),
              ),
              _divider(),
              Expanded(
                child: _summaryItem(
                  value: '$_disabled',
                  label: 'Disabled',
                  icon: Icons.lock_rounded,
                  color: const Color(0xFF9C4D4D),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _summaryItem({
    required String value,
    required String label,
    required IconData icon,
    required Color color,
  }) {
    return Column(
      children: [
        Icon(icon, color: color, size: 18),
        const SizedBox(height: 6),
        Text(
          value,
          style: GoogleFonts.dmSans(
            fontSize: 17,
            fontWeight: FontWeight.w700,
            color: OwnerFeaturePalette.text,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          label,
          textAlign: TextAlign.center,
          style: GoogleFonts.dmSans(
            fontSize: 9.5,
            color: OwnerFeaturePalette.textMuted,
          ),
        ),
      ],
    );
  }

  Widget _divider() {
    return Container(height: 42, width: 1, color: OwnerFeaturePalette.border);
  }

  Widget _buildSearch() {
    return Container(
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: OwnerFeaturePalette.border, width: 0.7),
      ),
      child: TextField(
        controller: _searchCtrl,
        onChanged: (_) => setState(() {}),
        style: GoogleFonts.dmSans(
          fontSize: 13.5,
          color: OwnerFeaturePalette.text,
        ),
        decoration: InputDecoration(
          hintText: 'Search name, email or username...',
          hintStyle: GoogleFonts.dmSans(
            color: OwnerFeaturePalette.textMuted,
            fontSize: 12.5,
          ),
          prefixIcon: Icon(
            Icons.search_rounded,
            color: OwnerFeaturePalette.textMuted,
            size: 20,
          ),
          suffixIcon: _searchCtrl.text.isEmpty
              ? null
              : IconButton(
                  onPressed: () {
                    _searchCtrl.clear();
                    setState(() {});
                  },
                  icon: const Icon(Icons.close_rounded, size: 18),
                ),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(vertical: 15),
        ),
      ),
    );
  }

  Widget _buildTypeFilters() {
    return SizedBox(
      height: 38,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: _types.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (context, index) {
          final type = _types[index];
          final selected = type == _selectedType;

          return GestureDetector(
            onTap: () {
              setState(() {
                _selectedType = type;
              });
            },
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 180),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: selected ? AppTheme.teal : OwnerFeaturePalette.surface,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: selected ? AppTheme.teal : OwnerFeaturePalette.border,
                ),
              ),
              child: Text(
                type,
                style: GoogleFonts.dmSans(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: selected
                      ? Colors.white
                      : OwnerFeaturePalette.textMuted,
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildSectionHeader() {
    return Row(
      children: [
        Expanded(
          child: Text(
            'Accounts',
            style: GoogleFonts.playfairDisplay(
              fontSize: 19,
              fontWeight: FontWeight.w700,
              color: OwnerFeaturePalette.text,
            ),
          ),
        ),
        Text(
          '${_filteredCredentials.length} found',
          style: GoogleFonts.dmSans(
            fontSize: 10.5,
            color: OwnerFeaturePalette.textMuted,
          ),
        ),
      ],
    );
  }

  Widget _buildCredentialsSliver() {
    final items = _filteredCredentials;

    if (items.isEmpty) {
      return SliverPadding(
        padding: const EdgeInsets.fromLTRB(18, 16, 18, 0),
        sliver: SliverToBoxAdapter(child: _buildEmptyState()),
      );
    }

    return SliverPadding(
      padding: const EdgeInsets.symmetric(horizontal: 18),
      sliver: SliverList(
        delegate: SliverChildBuilderDelegate((context, index) {
          return Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: _credentialCard(items[index]),
          );
        }, childCount: items.length),
      ),
    );
  }

  Widget _credentialCard(Map<String, dynamic> item) {
    final status = item['status'].toString();

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () => _openCredentialDetails(item),
        borderRadius: BorderRadius.circular(20),
        child: Container(
          padding: const EdgeInsets.all(15),
          decoration: BoxDecoration(
            color: OwnerFeaturePalette.surface,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: OwnerFeaturePalette.border, width: 0.6),
          ),
          child: Column(
            children: [
              Row(
                children: [
                  _avatar(item['name'].toString(), item['role'].toString()),

                  const SizedBox(width: 12),

                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Flexible(
                              child: Text(
                                item['name'].toString(),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: GoogleFonts.dmSans(
                                  fontSize: 14,
                                  fontWeight: FontWeight.w700,
                                  color: OwnerFeaturePalette.text,
                                ),
                              ),
                            ),
                            const SizedBox(width: 7),
                            _statusChip(status),
                          ],
                        ),

                        const SizedBox(height: 4),

                        Text(
                          '${item['role']} · ${item['email']}',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: GoogleFonts.dmSans(
                            fontSize: 10.5,
                            color: OwnerFeaturePalette.textMuted,
                          ),
                        ),
                      ],
                    ),
                  ),

                  const Icon(
                    Icons.chevron_right_rounded,
                    color: OwnerFeaturePalette.textMuted,
                  ),
                ],
              ),

              const SizedBox(height: 14),

              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: OwnerFeaturePalette.surfaceSoft,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: _credentialMini(
                        label: 'Username',
                        value: item['username'].toString(),
                        icon: Icons.person_outline_rounded,
                      ),
                    ),
                    Container(
                      width: 1,
                      height: 34,
                      color: OwnerFeaturePalette.border,
                    ),
                    Expanded(
                      child: _credentialMini(
                        label: 'Last Login',
                        value: item['lastLogin'].toString(),
                        icon: Icons.history_rounded,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _avatar(String name, String role) {
    final parts = name.trim().split(' ');

    String initials = parts.isNotEmpty && parts.first.isNotEmpty
        ? parts.first[0].toUpperCase()
        : 'U';

    if (parts.length > 1 && parts.last.isNotEmpty) {
      initials += parts.last[0].toUpperCase();
    }

    final color = _roleColor(role);

    return Container(
      width: 49,
      height: 49,
      decoration: BoxDecoration(
        color: color.withOpacity(0.10),
        shape: BoxShape.circle,
      ),
      child: Center(
        child: Text(
          initials,
          style: GoogleFonts.dmSans(
            fontSize: 14,
            fontWeight: FontWeight.w800,
            color: color,
          ),
        ),
      ),
    );
  }

  Color _roleColor(String role) {
    switch (role) {
      case 'Student':
        return const Color(0xFF7D55B4);
      case 'Parent':
        return const Color(0xFF3478B7);
      case 'Warden':
        return AppTheme.teal;
      default:
        return OwnerFeaturePalette.textMuted;
    }
  }

  Widget _statusChip(String status) {
    final color = _statusColor(status);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.10),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(
        status,
        style: GoogleFonts.dmSans(
          fontSize: 9,
          fontWeight: FontWeight.w700,
          color: color,
        ),
      ),
    );
  }

  Color _statusColor(String status) {
    switch (status) {
      case 'Active':
        return const Color(0xFF3D7B53);
      case 'Pending':
        return const Color(0xFFC07421);
      case 'Disabled':
        return const Color(0xFF9C4D4D);
      default:
        return OwnerFeaturePalette.textMuted;
    }
  }

  Widget _credentialMini({
    required String label,
    required String value,
    required IconData icon,
  }) {
    return Row(
      children: [
        Icon(icon, size: 16, color: AppTheme.teal),
        const SizedBox(width: 7),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                value,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.dmSans(
                  fontSize: 10.5,
                  fontWeight: FontWeight.w600,
                  color: OwnerFeaturePalette.text,
                ),
              ),
              const SizedBox(height: 1),
              Text(
                label,
                style: GoogleFonts.dmSans(
                  fontSize: 8.5,
                  color: OwnerFeaturePalette.textMuted,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildEmptyState() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 42),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: Column(
        children: [
          Container(
            width: 70,
            height: 70,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.07),
              shape: BoxShape.circle,
            ),
            child: Icon(
              Icons.password_outlined,
              size: 31,
              color: AppTheme.teal,
            ),
          ),

          const SizedBox(height: 16),

          Text(
            _loadingCredentials
                ? 'Loading database credentials…'
                : 'No credentials found',
            style: GoogleFonts.playfairDisplay(
              fontSize: 20,
              fontWeight: FontWeight.w700,
              color: OwnerFeaturePalette.text,
            ),
          ),

          const SizedBox(height: 6),

          Text(
            _loadingCredentials
                ? 'Fetching owner-created student accounts.'
                : 'Generate a student account to issue a one-time password.',
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(
              fontSize: 11.5,
              height: 1.5,
              color: OwnerFeaturePalette.textMuted,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFab() {
    return FloatingActionButton.extended(
      backgroundColor: AppTheme.teal,
      foregroundColor: Colors.white,
      elevation: 3,
      onPressed: _showGenerateCredentialSheet,
      icon: const Icon(Icons.add_rounded),
      label: Text(
        'Parent Login',
        style: GoogleFonts.dmSans(fontSize: 12, fontWeight: FontWeight.w700),
      ),
    );
  }

  void _openCredentialDetails(Map<String, dynamic> item) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) {
        bool showPassword = false;

        return StatefulBuilder(
          builder: (context, setSheetState) {
            final roleColor = _roleColor(item['role'].toString());

            return SafeArea(
              top: false,
              child: Container(
                margin: const EdgeInsets.all(12),
                padding: const EdgeInsets.fromLTRB(20, 12, 20, 24),
                decoration: BoxDecoration(
                  color: OwnerFeaturePalette.surfaceStrong,
                  borderRadius: BorderRadius.circular(28),
                ),
                child: SingleChildScrollView(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Center(
                        child: Container(
                          width: 42,
                          height: 4,
                          decoration: BoxDecoration(
                            color: OwnerFeaturePalette.border,
                            borderRadius: BorderRadius.circular(3),
                          ),
                        ),
                      ),

                      const SizedBox(height: 20),

                      Row(
                        children: [
                          _avatar(
                            item['name'].toString(),
                            item['role'].toString(),
                          ),

                          const SizedBox(width: 12),

                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  item['name'].toString(),
                                  style: GoogleFonts.playfairDisplay(
                                    fontSize: 21,
                                    fontWeight: FontWeight.w700,
                                    color: OwnerFeaturePalette.text,
                                  ),
                                ),
                                const SizedBox(height: 3),
                                Text(
                                  item['role'].toString(),
                                  style: GoogleFonts.dmSans(
                                    fontSize: 11,
                                    color: roleColor,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ],
                            ),
                          ),

                          _statusChip(item['status'].toString()),
                        ],
                      ),

                      const SizedBox(height: 22),

                      _credentialField(
                        label: 'Email',
                        value: item['email'].toString(),
                        icon: Icons.email_outlined,
                        onCopy: () =>
                            _copy(item['email'].toString(), 'Email copied'),
                      ),

                      const SizedBox(height: 10),

                      _credentialField(
                        label: 'Username',
                        value: item['username'].toString(),
                        icon: Icons.person_outline_rounded,
                        onCopy: () => _copy(
                          item['username'].toString(),
                          'Username copied',
                        ),
                      ),

                      const SizedBox(height: 10),

                      _credentialField(
                        label: 'Password',
                        value: showPassword
                            ? item['password'].toString()
                            : '••••••••',
                        icon: Icons.lock_outline_rounded,
                        trailing: IconButton(
                          onPressed: () {
                            setSheetState(() {
                              showPassword = !showPassword;
                            });
                          },
                          icon: Icon(
                            showPassword
                                ? Icons.visibility_off_outlined
                                : Icons.visibility_outlined,
                            size: 19,
                            color: OwnerFeaturePalette.textMuted,
                          ),
                        ),
                        onCopy: () => _copy(
                          item['password'].toString(),
                          'Password copied',
                        ),
                      ),

                      const SizedBox(height: 18),

                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: AppTheme.teal.withOpacity(0.06),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                            color: AppTheme.teal.withOpacity(0.15),
                          ),
                        ),
                        child: Row(
                          children: [
                            Icon(
                              Icons.history_rounded,
                              size: 18,
                              color: AppTheme.teal,
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                'Last login: ${item['lastLogin']}',
                                style: GoogleFonts.dmSans(
                                  fontSize: 11,
                                  color: AppTheme.teal,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),

                      const SizedBox(height: 20),

                      Row(
                        children: [
                          Expanded(
                            child: OutlinedButton.icon(
                              onPressed: () {
                                Navigator.pop(context);
                                _regenerateCredential(item);
                              },
                              icon: const Icon(Icons.refresh_rounded, size: 17),
                              label: const Text('Regenerate'),
                              style: OutlinedButton.styleFrom(
                                foregroundColor: AppTheme.teal,
                                side: const BorderSide(color: AppTheme.teal),
                                minimumSize: const Size.fromHeight(48),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(15),
                                ),
                              ),
                            ),
                          ),

                          const SizedBox(width: 10),

                          Expanded(
                            child: ElevatedButton.icon(
                              onPressed: () {
                                Navigator.pop(context);
                                _shareMessage(item);
                              },
                              icon: const Icon(Icons.share_outlined, size: 17),
                              label: const Text('Share'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: AppTheme.teal,
                                foregroundColor: Colors.white,
                                elevation: 0,
                                minimumSize: const Size.fromHeight(48),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(15),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),

                      const SizedBox(height: 12),

                      SizedBox(
                        width: double.infinity,
                        height: 48,
                        child: TextButton.icon(
                          onPressed: () {
                            Navigator.pop(context);

                            if (item['status'] == 'Disabled') {
                              _enableAccount(item);
                            } else {
                              _disableAccount(item);
                            }
                          },
                          icon: Icon(
                            item['status'] == 'Disabled'
                                ? Icons.lock_open_rounded
                                : Icons.lock_outline_rounded,
                          ),
                          label: Text(
                            item['status'] == 'Disabled'
                                ? 'Enable Account'
                                : 'Disable Account',
                          ),
                          style: TextButton.styleFrom(
                            foregroundColor: item['status'] == 'Disabled'
                                ? const Color(0xFF3D7B53)
                                : Colors.red.shade600,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(15),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            );
          },
        );
      },
    );
  }

  Widget _credentialField({
    required String label,
    required String value,
    required IconData icon,
    required VoidCallback onCopy,
    Widget? trailing,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 11),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: Row(
        children: [
          Icon(icon, size: 19, color: AppTheme.teal),

          const SizedBox(width: 11),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: GoogleFonts.dmSans(
                    fontSize: 9,
                    color: OwnerFeaturePalette.textMuted,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  value,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: OwnerFeaturePalette.text,
                  ),
                ),
              ],
            ),
          ),

          if (trailing != null) trailing,

          IconButton(
            tooltip: 'Copy',
            onPressed: onCopy,
            icon: Icon(
              Icons.copy_rounded,
              size: 18,
              color: OwnerFeaturePalette.textMuted,
            ),
          ),
        ],
      ),
    );
  }

  void _showGenerateCredentialSheet() {
    int? selectedBookingId;

    final nameCtrl = TextEditingController();
    final emailCtrl = TextEditingController();

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) {
        return StatefulBuilder(
          builder: (context, setSheetState) {
            return Padding(
              padding: EdgeInsets.only(
                bottom: MediaQuery.of(context).viewInsets.bottom,
              ),
              child: SafeArea(
                top: false,
                child: Container(
                  margin: const EdgeInsets.all(12),
                  padding: const EdgeInsets.fromLTRB(20, 12, 20, 24),
                  decoration: BoxDecoration(
                    color: OwnerFeaturePalette.surfaceStrong,
                    borderRadius: BorderRadius.circular(28),
                  ),
                  child: SingleChildScrollView(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Center(
                          child: Container(
                            width: 42,
                            height: 4,
                            decoration: BoxDecoration(
                              color: OwnerFeaturePalette.border,
                              borderRadius: BorderRadius.circular(3),
                            ),
                          ),
                        ),

                        const SizedBox(height: 20),

                        Text(
                          'Generate Parent Credentials',
                          style: GoogleFonts.playfairDisplay(
                            fontSize: 24,
                            fontWeight: FontWeight.w700,
                            color: OwnerFeaturePalette.text,
                          ),
                        ),

                        const SizedBox(height: 5),

                        Text(
                          'Select a student whose booking you have approved.',
                          style: GoogleFonts.dmSans(
                            fontSize: 11,
                            color: OwnerFeaturePalette.textMuted,
                          ),
                        ),

                        const SizedBox(height: 22),

                        Text(
                          'Approved student',
                          style: GoogleFonts.dmSans(
                            fontSize: 10.5,
                            fontWeight: FontWeight.w600,
                            color: OwnerFeaturePalette.textMuted,
                          ),
                        ),

                        const SizedBox(height: 8),

                        DropdownButtonFormField<int>(
                          value: selectedBookingId,
                          dropdownColor: Colors.white,
                          style: GoogleFonts.dmSans(
                            color: AppTheme.textPrimary,
                            fontSize: 13,
                          ),
                          decoration: InputDecoration(
                            filled: true,
                            fillColor: Colors.white,
                            hintText: _approvedStudents.isEmpty
                                ? 'No approved students available'
                                : 'Select approved student',
                            hintStyle: GoogleFonts.dmSans(
                              color: AppTheme.textMuted,
                            ),
                          ),
                          items: _approvedStudents
                              .map((student) {
                                final bookingId = int.tryParse(
                                  '${student['booking_id']}',
                                );
                                if (bookingId == null) return null;
                                return DropdownMenuItem<int>(
                                  value: bookingId,
                                  child: Text(
                                    '${student['name']} • ${student['parent_name']}',
                                  ),
                                );
                              })
                              .whereType<DropdownMenuItem<int>>()
                              .toList(),
                          onChanged: _approvedStudents.isEmpty
                              ? null
                              : (bookingId) {
                                  final student = _approvedStudents.firstWhere(
                                    (item) =>
                                        '${item['booking_id']}' == '$bookingId',
                                  );
                                  setSheetState(() {
                                    selectedBookingId = bookingId;
                                    nameCtrl.text =
                                        student['parent_name']?.toString() ??
                                        '';
                                    emailCtrl.text =
                                        student['parent_email']?.toString() ??
                                        '';
                                  });
                                },
                        ),

                        const SizedBox(height: 18),

                        _sheetField(
                          controller: nameCtrl,
                          hint: 'Parent full name',
                          icon: Icons.person_outline_rounded,
                        ),

                        const SizedBox(height: 12),

                        _sheetField(
                          controller: emailCtrl,
                          hint: 'Email address',
                          icon: Icons.email_outlined,
                          keyboardType: TextInputType.emailAddress,
                        ),

                        const SizedBox(height: 22),

                        SizedBox(
                          width: double.infinity,
                          height: 52,
                          child: ElevatedButton.icon(
                            onPressed: () async {
                              if (selectedBookingId == null) {
                                _showTemporary(
                                  'Select a confirmed student first.',
                                );
                                return;
                              }
                              if (nameCtrl.text.trim().isEmpty ||
                                  !RegExp(
                                    r'^[^\s@]+@[^\s@]+\.[^\s@]+$',
                                  ).hasMatch(emailCtrl.text.trim())) {
                                _showTemporary(
                                  'Enter the parent name and a valid email.',
                                );
                                return;
                              }

                              try {
                                final response = await Api()
                                    .issueOwnerParentCredential(
                                      selectedBookingId!,
                                      parentName: nameCtrl.text,
                                      parentEmail: emailCtrl.text,
                                    );
                                final name =
                                    response['parent_name']?.toString() ??
                                    nameCtrl.text.trim();
                                final email =
                                    response['parent_email']?.toString() ??
                                    emailCtrl.text.trim();
                                // The API returns `temporary_password`.  Reading a
                                // different key made a newly-created parent account
                                // look like an existing account, so its credentials
                                // were never shown to the owner.
                                final password = response['temporary_password']
                                    ?.toString();
                                await _loadCredentials();
                                if (!context.mounted) return;
                                Navigator.pop(context);
                                if (password == null || password.isEmpty) {
                                  _showTemporary(
                                    'Parent account already exists and is now linked to this student.',
                                  );
                                } else {
                                  _showGeneratedDialog(
                                    name: name,
                                    email: email,
                                    username: email.split('@').first,
                                    password: password,
                                    role: 'Parent',
                                  );
                                }
                              } catch (error) {
                                _showTemporary(
                                  error.toString().replaceFirst(
                                    'Exception: ',
                                    '',
                                  ),
                                );
                              }
                            },
                            style: ElevatedButton.styleFrom(
                              backgroundColor: AppTheme.teal,
                              foregroundColor: Colors.white,
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(16),
                              ),
                            ),
                            icon: const Icon(Icons.password_rounded, size: 18),
                            label: Text(
                              'Generate Parent Credentials',
                              style: GoogleFonts.dmSans(
                                fontSize: 14,
                                fontWeight: FontWeight.w700,
                              ),
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
      },
    );
  }

  Widget _sheetField({
    required TextEditingController controller,
    required String hint,
    required IconData icon,
    TextInputType? keyboardType,
    bool readOnly = false,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: TextField(
        controller: controller,
        keyboardType: keyboardType,
        readOnly: readOnly,
        style: GoogleFonts.dmSans(fontSize: 13, color: AppTheme.textPrimary),
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: GoogleFonts.dmSans(
            fontSize: 12.5,
            color: AppTheme.textMuted,
          ),
          prefixIcon: Icon(icon, size: 19, color: AppTheme.teal),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(vertical: 15),
        ),
      ),
    );
  }

  void _showGeneratedDialog({
    required String name,
    required String email,
    required String username,
    required String password,
    required String role,
  }) {
    showDialog(
      context: context,
      useRootNavigator: false,
      builder: (dialogContext) {
        return Dialog(
          backgroundColor: OwnerFeaturePalette.surfaceStrong,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(26),
          ),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 62,
                  height: 62,
                  decoration: BoxDecoration(
                    color: AppTheme.teal.withOpacity(0.08),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    Icons.check_circle_rounded,
                    color: AppTheme.teal,
                    size: 32,
                  ),
                ),

                const SizedBox(height: 16),

                Text(
                  'Credentials Created',
                  textAlign: TextAlign.center,
                  style: GoogleFonts.playfairDisplay(
                    fontSize: 21,
                    fontWeight: FontWeight.w700,
                    color: OwnerFeaturePalette.text,
                  ),
                ),

                const SizedBox(height: 5),

                Text(
                  '$role account for $name',
                  textAlign: TextAlign.center,
                  style: GoogleFonts.dmSans(
                    fontSize: 10.5,
                    color: OwnerFeaturePalette.textMuted,
                  ),
                ),

                const SizedBox(height: 18),

                _dialogCredential(
                  label: 'Username',
                  value: username,
                  onCopy: () => _copy(username, 'Username copied'),
                ),

                const SizedBox(height: 9),

                _dialogCredential(
                  label: 'Temporary Password',
                  value: password,
                  onCopy: () => _copy(password, 'Password copied'),
                ),

                const SizedBox(height: 18),

                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: const Color(0xFFFFF4E6),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Row(
                    children: [
                      const Icon(
                        Icons.warning_amber_rounded,
                        size: 18,
                        color: Color(0xFFC07421),
                      ),
                      const SizedBox(width: 9),
                      Expanded(
                        child: Text(
                          'Share this temporary password securely. The user should change it after first login.',
                          style: GoogleFonts.dmSans(
                            fontSize: 10,
                            height: 1.4,
                            color: const Color(0xFFC07421),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 18),

                SizedBox(
                  width: double.infinity,
                  height: 48,
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.pop(dialogContext);
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.teal,
                      foregroundColor: Colors.white,
                      elevation: 0,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(15),
                      ),
                    ),
                    child: Text(
                      'Done',
                      style: GoogleFonts.dmSans(fontWeight: FontWeight.w700),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _dialogCredential({
    required String label,
    required String value,
    required VoidCallback onCopy,
  }) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 11),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: GoogleFonts.dmSans(
                    fontSize: 9,
                    color: OwnerFeaturePalette.textMuted,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  value,
                  style: GoogleFonts.dmSans(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w700,
                    color: OwnerFeaturePalette.text,
                  ),
                ),
              ],
            ),
          ),

          IconButton(
            onPressed: onCopy,
            icon: Icon(Icons.copy_rounded, size: 18, color: AppTheme.teal),
          ),
        ],
      ),
    );
  }

  String _generateUsername(String name) {
    final clean = name.trim().toLowerCase().replaceAll(
      RegExp(r'[^a-z0-9 ]'),
      '',
    );

    final parts = clean
        .split(RegExp(r'\s+'))
        .where((e) => e.isNotEmpty)
        .toList();

    if (parts.isEmpty) {
      return 'staybuddy.user';
    }

    if (parts.length == 1) {
      return parts.first;
    }

    return '${parts.first}.${parts.last}';
  }

  String _generatePassword() {
    final timestamp = DateTime.now().millisecondsSinceEpoch;

    final code = (timestamp % 9000 + 1000).toString();

    return 'SB@$code';
  }

  Future<void> _copy(String value, String message) async {
    await Clipboard.setData(ClipboardData(text: value));

    if (!mounted) return;

    _showTemporary(message);
  }

  Future<void> _regenerateCredential(Map<String, dynamic> item) async {
    final id = int.tryParse('${item['userId']}'); if (id == null) return;
    try { final resets = await Api().resetOwnerUserPasswords([id]); if (resets.isEmpty) throw Exception('Password reset was not permitted'); setState(() { item['password'] = resets.first['temporary_password']; item['temporary'] = true; item['status'] = 'Active'; }); _showTemporary('Temporary password generated and previous sessions signed out.'); } catch (error) { _showTemporary(error.toString()); }
  }

  Future<void> _disableAccount(Map<String, dynamic> item) async {
    final id = int.tryParse('${item['userId']}'); if (id == null) return;
    try { await Api().setOwnerUserAccountStatus(id, false); setState(() => item['status'] = 'Disabled'); _showTemporary('${item['name']} account disabled and sessions revoked.'); } catch (error) { _showTemporary(error.toString()); }
  }

  Future<void> _enableAccount(Map<String, dynamic> item) async {
    final id = int.tryParse('${item['userId']}'); if (id == null) return;
    try { await Api().setOwnerUserAccountStatus(id, true); setState(() => item['status'] = 'Active'); _showTemporary('${item['name']} account enabled.'); } catch (error) { _showTemporary(error.toString()); }
  }

  void _shareMessage(Map<String, dynamic> item) {
    final text =
        '''
StayBuddy Login Credentials

Name: ${item['name']}
Role: ${item['role']}
Email: ${item['email']}
Username: ${item['username']}
Temporary Password: ${item['password']}
''';

    Clipboard.setData(ClipboardData(text: text));

    _showTemporary('Credentials copied. Native share will be connected later.');
  }

  void _showMoreMenu() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) {
        return SafeArea(
          top: false,
          child: Container(
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: OwnerFeaturePalette.surfaceStrong,
              borderRadius: BorderRadius.circular(26),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                _menuTile(
                  icon: Icons.refresh_rounded,
                  title: 'Bulk password reset',
                  onTap: () {
                    Navigator.pop(context);
                    _bulkPasswordReset();
                  },
                ),
                _menuTile(
                  icon: Icons.download_outlined,
                  title: 'Export credentials',
                  onTap: () {
                    Navigator.pop(context);
                    _exportCredentials();
                  },
                ),
                _menuTile(
                  icon: Icons.security_rounded,
                  title: 'Security settings',
                  onTap: () {
                    Navigator.pop(context);
                    _showTemporary(
                      'Security settings will be connected later.',
                    );
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _bulkPasswordReset() async {
    final users = _filteredCredentials.where((item) => item['userId'] != null && item['status'] != 'Disabled').toList();
    if (users.isEmpty) { _showTemporary('There are no active credentials to reset.'); return; }
    try {
      final ids = users.map((item) => int.tryParse('${item['userId']}')).whereType<int>().toList();
      final resets = await Api().resetOwnerUserPasswords(ids);
      final lines = resets.whereType<Map>().map((item) => 'User ${item['user_id']}: ${item['temporary_password']}').join('\n');
      await Clipboard.setData(ClipboardData(text: lines));
      _showTemporary('${resets.length} temporary passwords copied to the clipboard.');
      await _loadCredentials();
    } catch (error) { _showTemporary(error.toString()); }
  }

  Future<void> _exportCredentials() async {
    try {
      final data = await Api().getOwnerExportCsv('credentials');
      await Clipboard.setData(ClipboardData(text: data));
      _showTemporary('Credential CSV copied to the clipboard.');
    } catch (error) { _showTemporary(error.toString()); }
  }

  Widget _menuTile({
    required IconData icon,
    required String title,
    required VoidCallback onTap,
  }) {
    return ListTile(
      onTap: onTap,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      leading: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          color: AppTheme.teal.withOpacity(0.07),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Icon(icon, color: AppTheme.teal, size: 19),
      ),
      title: Text(
        title,
        style: GoogleFonts.dmSans(
          fontSize: 12.5,
          fontWeight: FontWeight.w600,
          color: OwnerFeaturePalette.text,
        ),
      ),
      trailing: const Icon(Icons.chevron_right_rounded),
    );
  }

  void _showTemporary(String message) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          backgroundColor: const Color(0xFF123F40),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
          content: Text(
            message,
            style: GoogleFonts.dmSans(color: Colors.white),
          ),
        ),
      );
  }
}
